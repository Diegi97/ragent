"""Delete explicitly named RAGent Turbopuffer namespaces."""

import argparse
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from ragent_core.retrievers.retriever import (
    DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
    create_turbopuffer_client,
)

logger = logging.getLogger(__name__)

_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete explicitly named physical Turbopuffer namespaces. Without "
            "--yes, only print the validated targets."
        )
    )
    parser.add_argument(
        "namespaces",
        nargs="+",
        help="Physical namespace names, for example ragent.test.catalog.",
    )
    parser.add_argument(
        "--namespace-prefix",
        default=os.getenv(
            "TURBOPUFFER_NAMESPACE_PREFIX",
            DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
        ),
        help="Required prefix for every target namespace.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the namespaces. Omit to preview targets.",
    )
    return parser.parse_args()


def _validated_targets(namespaces: Sequence[str], namespace_prefix: str) -> list[str]:
    if not namespace_prefix or not _NAMESPACE_PATTERN.fullmatch(namespace_prefix):
        raise ValueError(f"Invalid namespace prefix: {namespace_prefix!r}")

    required_prefix = f"{namespace_prefix}."
    targets: list[str] = []
    for namespace in dict.fromkeys(namespaces):
        if not _NAMESPACE_PATTERN.fullmatch(namespace):
            raise ValueError(f"Invalid Turbopuffer namespace: {namespace!r}")
        if not namespace.startswith(required_prefix):
            raise ValueError(
                f"Refusing namespace {namespace!r}; expected prefix "
                f"{required_prefix!r}."
            )
        targets.append(namespace)
    return targets


def delete_namespaces(
    namespaces: Sequence[str],
    *,
    namespace_prefix: str,
    confirmed: bool,
    client: Any = None,
) -> None:
    targets = _validated_targets(namespaces, namespace_prefix)
    if not confirmed:
        for namespace in targets:
            logger.info("Would delete Turbopuffer namespace %s", namespace)
        logger.info("Preview complete; rerun with --yes to delete these namespaces.")
        return

    resolved_client = client or create_turbopuffer_client()
    for namespace in targets:
        remote = resolved_client.namespace(namespace)
        if not remote.exists():
            logger.info("Namespace %s does not exist; skipping", namespace)
            continue
        remote.delete_all()
        logger.info("Deleted Turbopuffer namespace %s", namespace)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    try:
        delete_namespaces(
            args.namespaces,
            namespace_prefix=args.namespace_prefix,
            confirmed=args.yes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
