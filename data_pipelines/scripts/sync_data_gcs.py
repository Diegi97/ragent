#!/usr/bin/env python3
"""Mirror generated data between ``data/`` and Google Cloud Storage.

The local ``data/`` directory is the source for uploads; the configured GCS
prefix is the source for downloads. By default, ``gcloud storage rsync`` uses
``--delete-unmatched-destination-objects`` to remove files from the destination
that are not present in the source, producing an exact mirror. Pass
``--keep-extra`` to disable that behavior.

Examples
--------
# From data_pipelines/, upload every generated artifact under data/:
uv run python scripts/sync_data_gcs.py upload

# Preview the changes first:
uv run python scripts/sync_data_gcs.py upload --dry-run

# Restore all generated data from GCS:
uv run python scripts/sync_data_gcs.py download

Configuration
-------------
Set ``DATA_PIPELINES_GCS_URI`` to the destination prefix, for example
``gs://my-bucket/ragent/data-pipelines``. It may instead be passed with
``--gcs-uri``.

Set ``GCS_SERVICE_ACCOUNT`` to a service-account JSON path if explicit service
account activation is needed. Otherwise, the existing gcloud credentials are
used. This script requires ``gcloud`` on ``PATH``.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_DIR / "data"

# Use this project's environment file even when invoked from the repository root.
load_dotenv(PROJECT_DIR / ".env")

logger = logging.getLogger(__name__)


def _gcs_uri(cli_value: str | None) -> str:
    uri = cli_value or os.getenv("DATA_PIPELINES_GCS_URI")
    if not uri:
        raise ValueError(
            "DATA_PIPELINES_GCS_URI must be set or supplied with --gcs-uri."
        )
    uri = uri.rstrip("/")
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"GCS URI must have the form gs://bucket/prefix: {uri!r}")
    return uri


def _service_account_path() -> Path | None:
    value = os.getenv("GCS_SERVICE_ACCOUNT")
    if not value:
        return None
    return Path(value).expanduser()


def _ensure_tools() -> None:
    if shutil.which("gcloud") is None:
        raise EnvironmentError("Required tool not found on PATH: gcloud")


def _activate_service_account(key_path: Path | None) -> None:
    if key_path is None:
        return
    if not key_path.is_file():
        raise FileNotFoundError(
            f"GCS_SERVICE_ACCOUNT points to {str(key_path)!r}, which is not a file."
        )
    logger.info("Activating service account from %s", key_path)
    subprocess.run(
        ["gcloud", "auth", "activate-service-account", f"--key-file={key_path}"],
        check=True,
    )


def _rsync(
    source: str,
    destination: str,
    *,
    delete: bool,
    dry_run: bool,
) -> None:
    cmd = [
        "gcloud",
        "storage",
        "rsync",
        source,
        destination,
        "--recursive",
        "--no-ignore-symlinks",
    ]
    if delete:
        cmd.append("--delete-unmatched-destination-objects")
    if dry_run:
        cmd.append("--dry-run")
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def upload(gcs_uri: str, *, delete: bool = True, dry_run: bool = False) -> None:
    """Mirror the complete local data directory to GCS."""
    if not LOCAL_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Local data directory does not exist: {LOCAL_DATA_DIR}"
        )
    if not any(LOCAL_DATA_DIR.iterdir()):
        raise ValueError(
            f"Refusing to mirror an empty local data directory: {LOCAL_DATA_DIR}"
        )

    key_path = _service_account_path()
    _ensure_tools()
    _activate_service_account(key_path)
    logger.info("Uploading all generated data: %s -> %s", LOCAL_DATA_DIR, gcs_uri)
    _rsync(str(LOCAL_DATA_DIR), gcs_uri, delete=delete, dry_run=dry_run)


def download(gcs_uri: str, *, delete: bool = True, dry_run: bool = False) -> None:
    """Mirror the complete GCS data prefix to the local data directory."""
    key_path = _service_account_path()
    _ensure_tools()
    _activate_service_account(key_path)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading all generated data: %s -> %s", gcs_uri, LOCAL_DATA_DIR)
    _rsync(gcs_uri, str(LOCAL_DATA_DIR), delete=delete, dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror data_pipelines/data between local disk and GCS."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--gcs-uri",
        help="GCS destination prefix (overrides DATA_PIPELINES_GCS_URI)",
    )
    common.add_argument(
        "--keep-extra",
        action="store_true",
        help="do not delete destination files that are absent from the source",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="show the changes gcloud storage would make without applying them",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("upload", parents=[common], help="local data -> GCS")
    subparsers.add_parser("download", parents=[common], help="GCS -> local data")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    try:
        gcs_uri = _gcs_uri(args.gcs_uri)
        kwargs = {"delete": not args.keep_extra, "dry_run": args.dry_run}
        if args.command == "upload":
            upload(gcs_uri, **kwargs)
        else:
            download(gcs_uri, **kwargs)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
