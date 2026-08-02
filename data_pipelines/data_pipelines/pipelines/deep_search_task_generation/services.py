import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from ragent_core.data_sources import load_corpus


async def chat_completion(messages: Sequence[dict[str, str]], model: str) -> str:
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "5")),
    )
    async with client:
        response = await client.chat.completions.create(
            model=model,
            messages=list(messages),
        )
    return response.choices[0].message.content if response.choices else ""


def load_dataset(data_source: str) -> tuple[Any, str | None, str | None]:
    return load_corpus(data_source)


def _fireworks_credentials() -> tuple[str, str]:
    account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
    api_key = os.getenv("OPENAI_API_KEY")
    if not account_id:
        raise ValueError("FIREWORKS_ACCOUNT_ID is not set")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return account_id, api_key


def _raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        message = f"{exc}\nResponse body: {body}" if body else str(exc)
        raise httpx.HTTPStatusError(
            message,
            request=exc.request,
            response=exc.response,
        ) from exc


def upload_batch_dataset(
    jsonl_path: Path,
    dataset_name: str,
    timeout: float,
) -> dict[str, Any]:
    account_id, api_key = _fireworks_credentials()
    create_url = f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets"
    upload_url = f"{create_url}/{dataset_name}:upload"
    headers = {"Authorization": f"Bearer {api_key}"}
    with jsonl_path.open(encoding="utf-8") as fp:
        example_count = sum(1 for line in fp if line.strip())
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            create_url,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "datasetId": dataset_name,
                "dataset": {"userUploaded": {}, "example_count": example_count},
            },
        )
        if response.status_code == 409:
            payload: dict[str, Any] = {"name": dataset_name}
        else:
            _raise_for_status_with_body(response)
            payload = response.json()
        with jsonl_path.open("rb") as fp:
            upload = client.post(
                upload_url,
                headers=headers,
                files={"file": (jsonl_path.name, fp, "application/jsonl")},
            )
        _raise_for_status_with_body(upload)
    return upload.json() or payload


def download_batch_dataset(
    dataset_name: str,
    output_directory: Path,
    timeout: float,
) -> list[Path]:
    account_id, api_key = _fireworks_credentials()
    endpoint = (
        f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets/"
        f"{dataset_name}:getDownloadEndpoint"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=timeout) as client:
        response = client.get(endpoint, headers=headers)
        _raise_for_status_with_body(response)
        urls = response.json().get("filenameToSignedUrls") or {}
        downloaded: list[Path] = []
        used_names: set[str] = set()
        for index, (object_path, signed_url) in enumerate(urls.items()):
            filename = Path(object_path).name or f"output-{index:04d}.jsonl"
            if filename in used_names:
                filename = f"{index:04d}-{filename}"
            used_names.add(filename)
            destination = output_directory / filename
            with client.stream("GET", signed_url) as stream:
                stream.raise_for_status()
                with destination.open("wb") as fp:
                    for chunk in stream.iter_bytes():
                        fp.write(chunk)
            downloaded.append(destination)
    return downloaded
