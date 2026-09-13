"""Download the public JSON documents used by the LLIS website, then normalize them."""

import argparse
import json
import logging
import re
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from datasets import Dataset, DatasetInfo
from httpx_retries import Retry, RetryTransport

from ragent_core.data_sources import filter_by_word_count

from . import DATA_DIR, DESCRIPTION

logger = logging.getLogger(__name__)
BASE_URL = "https://llis.nasa.gov"
SEARCH_URL = f"{BASE_URL}/llis/lesson/_search"
BODY_FIELDS = (
    ("Abstract", ("lessonAbstract",)),
    ("Driving Event", ("drivingEvent", "description_event")),
    ("Lesson(s) Learned", ("lesson", "lesson_learned")),
    ("Recommendation(s)", ("recommendation",)),
    ("Evidence of Recurrence Control Effectiveness", ("evidence",)),
)
METADATA_FIELDS = (
    ("Date Lesson Occurred", ("lesson_date", "lessonDate")),
    ("Submitting Organization", ("organization",)),
    ("Program Relation", ("programRelation",)),
    ("Program/Project Phase", ("programPhase",)),
    ("Program Role", ("programRole",)),
    ("Mission Directorate(s)", ("missionDirectorate",)),
    ("Topic(s)", ("categories",)),
    ("Related Policy", ("relatedPolicy",)),
)


def _url(value: str) -> str:
    url = urljoin(BASE_URL, value)
    return url if urlsplit(url).scheme in {"http", "https"} else ""


class _LessonHTMLParser(HTMLParser):
    """Keep prose, list/table boundaries, and references without executing HTML."""

    BLOCKS = {"p", "div", "section", "blockquote", "ul", "ol", "table", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.ignored: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.ignored:
            return
        attributes = dict(attrs)
        if tag in {"script", "style"}:
            self.ignored = tag
        elif tag in self.BLOCKS or re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "a":
            href = attributes.get("href")
            self.links.append(_url(href) if href else "")
        elif tag == "img" and attributes.get("alt"):
            self.parts.append(f" [Image: {attributes['alt']}] ")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored:
            if tag == self.ignored:
                self.ignored = None
            return
        if tag == "a" and self.links:
            url = self.links.pop()
            if url:
                self.parts.append(f" ({url})")
        elif tag in self.BLOCKS or re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def clean_text(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("name"))
    if isinstance(value, list):
        return ", ".join(text for item in value if (text := clean_text(item)))
    if not isinstance(value, str):
        return ""
    parser = _LessonHTMLParser()
    parser.feed(value)
    parser.close()
    text = "\n".join(
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in "".join(parser.parts).splitlines()
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "" if text.casefold() in {"none", "n/a", "null"} else text


def _field(source: dict[str, Any], aliases: tuple[str, ...]) -> str:
    return next((text for key in aliases if (text := clean_text(source.get(key)))), "")


def normalize_lesson(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit["_source"]
    # NASA numbers are stable unsigned document IDs; never renumber filtered rows.
    number = str(source["lesson_number"])
    if not number.isascii() or not number.isdecimal() or int(number) < 1:
        raise ValueError(f"Invalid NASA lesson number: {number!r}")
    lesson_id = int(number)
    if str(hit["_id"]) != str(lesson_id):
        raise ValueError(f"NASA lesson number does not match record ID: {number}")
    title = clean_text(source.get("title")) or f"NASA Lesson {lesson_id}"
    sections = [
        f"## {label}\n\n{text}"
        for label, aliases in BODY_FIELDS
        if (text := _field(source, aliases))
    ]
    if not sections:
        raise ValueError(f"NASA lesson {lesson_id} has no substantive content")
    metadata = [
        f"NASA Lesson ID: {lesson_id}",
        f"Source: {BASE_URL}/lesson/{lesson_id}",
    ]
    metadata.extend(
        f"{label}: {text}"
        for label, aliases in METADATA_FIELDS
        if (text := _field(source, aliases))
    )
    attachments = []
    for attachment in source.get("attachments") or []:
        link = _url(attachment.get("link", "")) if attachment.get("link") else ""
        if link:
            attachments.append(
                f"- {clean_text(attachment.get('name')) or 'Attachment'}: {link}"
            )
    for key, value in source.items():
        if re.fullmatch(r"documentUrl\d+", key) and value and (link := _url(value)):
            index = key.removeprefix("documentUrl")
            name = clean_text(source.get(f"documentDescription{index}"))
            name = (
                name or clean_text(source.get(f"documentName{index}")) or "Attachment"
            )
            attachments.append(f"- {name}: {link}")
    if attachments:
        sections.append(
            "## Attachments (links only)\n\n" + "\n".join(dict.fromkeys(attachments))
        )
    return {
        "id": lesson_id,
        "title": title,
        "text": "\n\n".join([f"# {title}", "\n".join(metadata), *sections]),
    }


def fetch_lessons(
    client: httpx.Client, *, page_size: int = 100, delay: float = 1.0
) -> Iterator[dict[str, Any]]:
    """Read every numbered lesson; reject partial or changing search results."""
    if not 1 <= page_size <= 1000 or delay < 0:
        raise ValueError("page_size must be 1–1000 and delay must be nonnegative")
    offset = 0
    total = None
    seen: set[str] = set()
    while total is None or offset < total:
        if offset:
            time.sleep(delay)
        response = client.post(
            SEARCH_URL,
            json={
                # The public index also contains records that are not lessons.
                "query": {"exists": {"field": "lesson_number"}},
                "sort": [{"lesson_number": "asc"}],
                "from": offset,
                "size": page_size,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("timed_out") or payload.get("_shards", {}).get("failed", 0):
            raise ValueError("NASA LLIS returned incomplete search results")
        count = payload["hits"]["total"]
        if isinstance(count, dict):
            if count.get("relation") != "eq":
                raise ValueError("NASA LLIS did not return an exact document count")
            count = count["value"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("NASA LLIS returned an invalid or empty document count")
        if total is not None and count != total:
            raise ValueError("NASA LLIS changed during download; retry the snapshot")
        total = count
        hits = payload["hits"]["hits"]
        if not hits or len(hits) > min(page_size, total - offset):
            raise ValueError("NASA LLIS returned an inconsistent page")
        for hit in hits:
            record_id = str(hit["_id"])
            if record_id in seen:
                raise ValueError(f"NASA LLIS returned duplicate lesson {record_id}")
            seen.add(record_id)
            yield hit
        offset += len(hits)
        logger.info("Downloaded %s/%s NASA lessons", offset, total)


def prepare_dataset(output_dir: Path = DATA_DIR) -> Path:
    """Publish a local cache only after a complete, validated download."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; use a new --output-dir")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        allowed_methods={"POST"},  # This endpoint is a read-only search.
        status_forcelist={429, 500, 502, 503, 504},
    )
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        records = []
        with (
            httpx.Client(
                transport=RetryTransport(retry=retry),
                timeout=60.0,
                headers={"User-Agent": "ragent-nasa-lessons-learned/1.0"},
            ) as client,
            (staging / "raw.jsonl").open("w", encoding="utf-8") as raw,
        ):
            for hit in fetch_lessons(client):
                raw.write(json.dumps(hit, ensure_ascii=False) + "\n")
                records.append(normalize_lesson(hit))
        records.sort(key=lambda row: row["id"])
        dataset = Dataset.from_list(
            records,
            info=DatasetInfo(
                dataset_name="nasa_lessons_learned", description=DESCRIPTION
            ),
        )
        dataset = filter_by_word_count(dataset)
        if not len(dataset):
            raise ValueError("No NASA lessons passed the document length filter")
        dataset.save_to_disk(str(staging / "dataset"))
        dataset.to_json(str(staging / "train.jsonl"), force_ascii=False)
        filtered = [
            {"id": row["id"], "word_count": len(row["text"].split())}
            for row in records
            if not 50 <= len(row["text"].split()) <= 8000
        ]
        manifest = {
            "source_url": BASE_URL,
            "search_url": SEARCH_URL,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "downloaded_documents": len(records),
            "retained_documents": len(dataset),
            "filtered_documents": len(records) - len(dataset),
            "excluded_lessons": filtered,
            "min_words": 50,
            "max_words": 8000,
            "attachments_downloaded": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    logger.info(
        "Saved %s/%s NASA lessons to %s (50–8000 words; raw records retained)",
        len(dataset),
        len(records),
        output_dir,
    )
    return output_dir / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_dataset(args.output_dir)


if __name__ == "__main__":
    main()
