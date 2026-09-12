import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx
from prefect import task

from data_pipelines.artifacts.inputs import count_jsonl
from data_pipelines.providers.config import OPENAI_API_KEY_ENV
from ragent_core.provider_errors import ProviderError, ProviderFailureKind

FIREWORKS_RESOURCE_ID_MAX_LENGTH = 63
FIREWORKS_DATASETS_URL = "https://api.fireworks.ai/v1/accounts/{account_id}/datasets"

FACT_DATASET_PREFIX = "deep-search-tasks-"


class FireworksError(ProviderError):
    def __init__(self, kind: ProviderFailureKind, status_code: int | None = None):
        super().__init__("Fireworks", kind, status_code)


@dataclass(frozen=True)
class FireworksUploadResult:
    dataset_name: str
    example_count: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def upload_batch_dataset(
    jsonl_path: Path,
    dataset_name: str,
    timeout: float,
) -> FireworksUploadResult:
    account_id, api_key = _fireworks_credentials()
    create_url = FIREWORKS_DATASETS_URL.format(account_id=account_id)
    upload_url = f"{create_url}/{dataset_name}:upload"
    headers = {"Authorization": f"Bearer {api_key}"}
    example_count = count_jsonl(jsonl_path)
    with _provider_client(timeout) as client:
        response = client.post(
            create_url,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "datasetId": dataset_name,
                "dataset": {"userUploaded": {}, "exampleCount": str(example_count)},
            },
        )
        if response.status_code == 409:
            raise FileExistsError(
                f"Fireworks dataset {dataset_name!r} already exists; its ownership is unverified"
            )
        _require_success(response)
        _response_object(response)
        with jsonl_path.open("rb") as fp:
            upload = client.post(
                upload_url,
                headers=headers,
                files={"file": (jsonl_path.name, fp, "application/jsonl")},
            )
        _require_success(upload)
        _response_object(upload)
    return FireworksUploadResult(dataset_name, example_count)


def download_batch_dataset(
    dataset_name: str,
    output_directory: Path,
    timeout: float,
) -> list[Path]:
    account_id, api_key = _fireworks_credentials()
    collection_url = FIREWORKS_DATASETS_URL.format(account_id=account_id)
    endpoint = f"{collection_url}/{dataset_name}:getDownloadEndpoint"
    headers = {"Authorization": f"Bearer {api_key}"}
    with _provider_client(timeout) as client:
        response = client.get(endpoint, headers=headers)
        _require_success(response)
        metadata = _response_object(response)
        signed_urls_by_object_path = metadata.get("filenameToSignedUrls")
        if (
            not isinstance(signed_urls_by_object_path, dict)
            or not signed_urls_by_object_path
        ):
            raise FireworksError(ProviderFailureKind.INVALID_RESPONSE)
        for object_path, signed_url in signed_urls_by_object_path.items():
            if (
                not isinstance(object_path, str)
                or not object_path.strip()
                or not isinstance(signed_url, str)
            ):
                raise FireworksError(ProviderFailureKind.INVALID_RESPONSE)
            parsed_url = urlsplit(signed_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise FireworksError(ProviderFailureKind.INVALID_RESPONSE)
        downloaded: list[Path] = []
        used_names: set[str] = set()
        with TemporaryDirectory(prefix=".fireworks-", dir=output_directory) as staging:
            staged: list[tuple[Path, Path]] = []
            for index, (object_path, signed_url) in enumerate(
                signed_urls_by_object_path.items()
            ):
                basename = Path(object_path).name
                if basename in {"", ".", ".."}:
                    raise FireworksError(ProviderFailureKind.INVALID_RESPONSE)
                # Provider paths can share a basename; bound the local name and keep each object.
                basename = basename.encode("utf-8")[:200].decode(
                    "utf-8", errors="ignore"
                )
                filename = basename
                suffix = index
                while filename in used_names:
                    filename = f"{suffix:04d}-{basename}"
                    suffix += 1
                used_names.add(filename)
                temporary = Path(staging) / filename
                destination = output_directory / filename
                with client.stream("GET", signed_url) as stream:
                    _require_success(stream)
                    with temporary.open("wb") as fp:
                        for chunk in stream.iter_bytes():
                            fp.write(chunk)
                staged.append((temporary, destination))
            for temporary, destination in staged:
                temporary.replace(destination)
                downloaded.append(destination)

    return downloaded


@task(name="download-fact-extraction-output", retries=0, persist_result=False)
async def download_output(
    dataset_name: str, output_directory: Path, timeout: float
) -> list[Path]:
    return await anyio.to_thread.run_sync(
        download_batch_dataset, dataset_name, output_directory, timeout
    )


def fact_dataset_name(data_source: str, run_id: str) -> str:
    identity = uuid.UUID(run_id).hex
    remaining = (
        FIREWORKS_RESOURCE_ID_MAX_LENGTH - len(FACT_DATASET_PREFIX) - len(identity) - 1
    )
    slug = re.sub(r"[^a-z0-9-]+", "-", data_source.lower()).strip("-") or "source"
    return f"{FACT_DATASET_PREFIX}{slug[:remaining].rstrip('-')}-{identity}"


def _fireworks_credentials() -> tuple[str, str]:
    account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not account_id:
        raise ValueError("FIREWORKS_ACCOUNT_ID is not set")
    if not api_key:
        raise ValueError(f"{OPENAI_API_KEY_ENV} is not set")
    return account_id, api_key


def _require_success(response: httpx.Response) -> None:
    if not response.is_success:
        raise FireworksError(
            ProviderFailureKind.for_status(response.status_code), response.status_code
        )


@contextmanager
def _provider_client(timeout: float) -> Iterator[httpx.Client]:
    try:
        with httpx.Client(timeout=timeout) as client:
            yield client
    except httpx.RequestError as exc:
        raise FireworksError(ProviderFailureKind.TRANSPORT) from exc


def _response_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise FireworksError(ProviderFailureKind.INVALID_RESPONSE) from exc
    if not isinstance(payload, dict):
        raise FireworksError(ProviderFailureKind.INVALID_RESPONSE)
    return payload
