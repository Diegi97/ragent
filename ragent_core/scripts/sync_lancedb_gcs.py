"""Mirror LanceDB namespaces between local disk and GCS via ``gsutil rsync``.

GCS is the durable source of truth; local disk is the hot read path used by
:class:`ragent_core.retrievers.LanceDBRetriever` (which never touches GCS).
Use this script to push a freshly built local index up to GCS, or to pull a
namespace down to local disk before running evals/training.

Examples
--------
# Build locally, then upload the namespace to GCS (durable store):
uv run --project ragent_core python scripts/build_lancedb_indexes.py --data-source nampdn_ai_devdocs_io --device cuda
uv run --project ragent_core python scripts/sync_lancedb_gcs.py upload --namespace default

# On a worker: download the namespace to local disk, then read locally:
uv run --project ragent_core python scripts/sync_lancedb_gcs.py download --namespace default
uv run --project ragent_core python playground.py

Authentication
--------------
Set ``GCS_SERVICE_ACCOUNT`` (path to a service account JSON) in the
environment; the script activates it with ``gcloud auth activate-service-account``
before running ``gsutil rsync``. If you are already authenticated (e.g. via
``gcloud auth application-default login``), leave it unset.

Requires ``gsutil`` and (when using a service account) ``gcloud`` on ``PATH``.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv

from ragent_core.config import GCS_SERVICE_ACCOUNT, LANCEDB_GCS_URI
from ragent_core.retrievers.retriever import DEFAULT_LOCAL_DB_URI

logger = logging.getLogger(__name__)


load_dotenv()


def _local_namespace_dir(namespace: str) -> str:
    return os.path.join(DEFAULT_LOCAL_DB_URI.rstrip("/"), namespace)


def _gcs_namespace_uri(namespace: str) -> str:
    if not LANCEDB_GCS_URI:
        raise ValueError(
            "LANCEDB_GCS_URI must be set in the environment to sync with GCS."
        )
    return f"{LANCEDB_GCS_URI.rstrip('/')}/{namespace}"


def _ensure_tools(require_gcloud: bool) -> None:
    missing = []
    if shutil.which("gsutil") is None:
        missing.append("gsutil")
    if require_gcloud and shutil.which("gcloud") is None:
        missing.append("gcloud")
    if missing:
        raise EnvironmentError(
            "Required tool(s) not found on PATH: " + ", ".join(missing)
        )


def _activate_service_account() -> None:
    """Activate the service account for gsutil if one is configured."""
    if not GCS_SERVICE_ACCOUNT:
        return
    key_path = GCS_SERVICE_ACCOUNT
    if not os.path.isfile(key_path):
        raise FileNotFoundError(
            f"GCS_SERVICE_ACCOUNT points to '{key_path}', which is not a file."
        )
    logger.info("Activating service account from %s", key_path)
    subprocess.run(
        ["gcloud", "auth", "activate-service-account", f"--key-file={key_path}"],
        check=True,
    )


def _rsync(src: str, dst: str, *, delete: bool = True) -> None:
    """Run ``gsutil -m rsync -r [-d] src dst`` making dst mirror src."""
    cmd = ["gsutil", "-m", "rsync", "-r"]
    if delete:
        cmd.append("-d")
    cmd.extend([src, dst])
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def upload(namespace: str) -> None:
    """Push a local namespace up to GCS (local is the source of truth here)."""
    _ensure_tools(require_gcloud=bool(GCS_SERVICE_ACCOUNT))
    local_dir = _local_namespace_dir(namespace)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"Local namespace directory does not exist: {local_dir}. "
            "Build the index first with scripts/build_lancedb_indexes.py."
        )
    _activate_service_account()
    gcs_uri = _gcs_namespace_uri(namespace)
    logger.info(
        "Uploading LanceDB namespace '%s': %s -> %s", namespace, local_dir, gcs_uri
    )
    _rsync(local_dir, gcs_uri)


def download(namespace: str) -> None:
    """Pull a namespace down from GCS to local disk (GCS is the source of truth)."""
    _ensure_tools(require_gcloud=bool(GCS_SERVICE_ACCOUNT))
    _activate_service_account()
    gcs_uri = _gcs_namespace_uri(namespace)
    local_dir = _local_namespace_dir(namespace)
    os.makedirs(local_dir, exist_ok=True)
    logger.info(
        "Downloading LanceDB namespace '%s': %s -> %s", namespace, gcs_uri, local_dir
    )
    _rsync(gcs_uri, local_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror LanceDB namespaces between local disk and GCS."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", default="default")
    sub.add_parser("upload", parents=[common], help="local disk -> GCS")
    sub.add_parser("download", parents=[common], help="GCS -> local disk")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    try:
        if args.command == "upload":
            upload(args.namespace)
        elif args.command == "download":
            download(args.namespace)
        else:  # pragma: no cover - argparse enforces the choice
            parser_error()
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        logger.error("%s", exc)
        sys.exit(1)


def parser_error() -> None:
    raise SystemExit("Unknown command. Use 'upload' or 'download'.")


if __name__ == "__main__":
    main()
