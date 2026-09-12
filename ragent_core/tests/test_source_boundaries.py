from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import pytest
from datasets import Dataset

from ragent_core import data_sources
from ragent_core.data_sources import repository
from ragent_core.data_sources.records import CORE_COLUMNS
from ragent_core.data_sources.repository import RepositoryCorpus, extract_markdown_files


def test_legacy_tuple_normalization():
    dataset = Dataset.from_list([{"id": 0, "text": "a"}])
    spec = data_sources.DataSourceSpec.from_loader_result((dataset, "description"))
    assert spec.dataset is dataset
    assert spec.name is None
    assert spec.description == "description"


@pytest.mark.parametrize(
    "error", [ImportError("dependency failed"), AttributeError("malformed loader")]
)
def test_custom_loader_failure_never_falls_back(monkeypatch, error):
    def fail():
        raise error

    monkeypatch.setattr(data_sources, "get_data_source_loader", lambda _: fail)
    with patch.object(data_sources, "load_dataset") as fallback:
        with pytest.raises(type(error), match=str(error)):
            data_sources.load_corpus("custom")
        fallback.assert_not_called()


def test_missing_custom_module_uses_huggingface(monkeypatch):
    dataset = Dataset.from_list([{"id": 0, "text": "a"}])

    def missing(_):
        raise ModuleNotFoundError(name="ragent_core.data_sources.external_corpus")

    monkeypatch.setattr(data_sources, "get_data_source_loader", missing)
    monkeypatch.setattr(
        data_sources, "load_dataset", lambda *args, **kwargs: {"train": dataset}
    )
    assert data_sources.load_corpus("external/corpus")[0].to_list() == [
        {"id": 0, "text": "a", "title": ""}
    ]


def test_extraction_read_error_is_not_a_partial_success(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("# Title\nbody")
    with pytest.raises(FileNotFoundError):
        extract_markdown_files(tmp_path, [good, tmp_path / "missing.md"])


def test_failed_rebuild_preserves_previous_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def clone(command, **kwargs):
        content = Path(command[-1]) / "content"
        content.mkdir()
        (content / "a.md").write_text("# Title\nbody")

    monkeypatch.setattr("ragent_core.data_sources.repository.subprocess.run", clone)
    corpus = RepositoryCorpus(
        "source",
        "https://example.test/repo",
        "content",
        lambda path: extract_markdown_files(path, path.glob("*.md")),
    )
    assert len(corpus.load()) == 1

    def fail(_):
        raise OSError("read failed")

    failed = RepositoryCorpus(
        corpus.name, corpus.repository_url, corpus.content_directory, fail
    )
    with pytest.raises(OSError, match="read failed"):
        failed.prepare()
    assert corpus.load()[0]["title"] == "Title"


def test_concurrent_first_load_builds_once(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    builds = []
    guard = Lock()

    def clone(command, **kwargs):
        with guard:
            builds.append(command)
        content = Path(command[-1]) / "content"
        content.mkdir()
        (content / "a.md").write_text("# Title\nbody")

    monkeypatch.setattr("ragent_core.data_sources.repository.subprocess.run", clone)
    corpus = RepositoryCorpus(
        "source",
        "https://example.test/repo",
        "content",
        lambda path: extract_markdown_files(path, path.glob("*.md")),
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        datasets = list(pool.map(lambda _: corpus.load(), range(4)))
    assert len(builds) == 1
    assert all(dataset[0]["title"] == "Title" for dataset in datasets)


def test_failed_directory_swap_rolls_back_previous_cache(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    title = "Original"

    def clone(command, **kwargs):
        content = Path(command[-1]) / "content"
        content.mkdir()
        (content / "a.md").write_text(f"# {title}\nbody")

    monkeypatch.setattr(repository.subprocess, "run", clone)
    corpus = RepositoryCorpus(
        "source",
        "https://example.test/repo",
        "content",
        lambda path: extract_markdown_files(path, path.glob("*.md")),
    )
    assert corpus.load()[0]["title"] == title
    real_replace = repository.os.replace

    def fail_staging_swap(source, destination):
        if Path(destination) == corpus.directory and not Path(source).name.startswith(
            ".source.backup-"
        ):
            raise OSError("staging swap failed")
        return real_replace(source, destination)

    title = "Replacement"
    monkeypatch.setattr(repository.os, "replace", fail_staging_swap)
    with pytest.raises(OSError, match="staging swap failed"):
        corpus.prepare()
    assert corpus.load()[0]["title"] == "Original"
    assert {path.name for path in corpus.directory.parent.iterdir()} - {
        ".source.lock"
    } == {"source"}


@pytest.mark.parametrize("id_column, expected_id", [(None, 1), ("id", 20)])
def test_source_normalization_preserves_identity_before_filtering(
    id_column, expected_id
):
    dataset = Dataset.from_list(
        [
            {"id": 10, "title": "Short", "text": "too short", "extra": "drop"},
            {"id": 20, "title": "Title", "text": "word " * 60, "extra": "drop"},
        ]
    )
    normalized = data_sources.normalize_source_dataset(dataset, id_column=id_column)
    assert len(normalized) == 1
    assert normalized[0]["id"] == expected_id
    assert normalized[0]["text"].startswith("# Title\n\n")
    assert set(normalized.column_names) == set(CORE_COLUMNS)


@pytest.mark.parametrize(
    "row", [{"text": "body"}, {"id": True, "text": "body"}, {"id": 0, "text": 17}]
)
def test_corpus_loader_rejects_malformed_rows_before_consumers(monkeypatch, row):
    monkeypatch.setattr(
        data_sources,
        "get_data_source_loader",
        lambda _: lambda: Dataset.from_list([row]),
    )
    with pytest.raises((ValueError, TypeError)):
        data_sources.load_corpus("malformed")
