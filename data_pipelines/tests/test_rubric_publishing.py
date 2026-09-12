import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from data_pipelines.publishing import question_rubrics as publisher
from data_pipelines.publishing.question_rubrics import cli
from data_pipelines.publishing.question_rubrics.datasets import (
    SPLIT_NAMES,
    load_question_rubrics,
    merge_datasets,
    split_dataset,
)
from data_pipelines.publishing.question_rubrics.schema import (
    QUESTION_RUBRIC_FEATURES,
    align_existing_split,
)

TRAIN_SPLIT, TEST_SPLIT = SPLIT_NAMES


def row(question, source="source"):
    return {
        "entity": "Entity",
        "question": question,
        "rubric": [{"criterion": "Requirement", "doc_ids": [0]}],
        "doc_ids": [0],
        "evolution_strategies": [],
        "data_source": source,
    }


def splits(train=(), test=()):
    return DatasetDict(
        {
            name: Dataset.from_list(list(records), features=QUESTION_RUBRIC_FEATURES)
            if records
            else Dataset.from_dict(
                {column: [] for column in QUESTION_RUBRIC_FEATURES},
                features=QUESTION_RUBRIC_FEATURES,
            )
            for name, records in [(TRAIN_SPLIT, train), (TEST_SPLIT, test)]
        }
    )


def test_replayed_batch_keeps_original_assignments():
    first = splits([row("train question")], [row("test question")])
    replay = splits([row("test question")], [row("train question")])
    merged = merge_datasets(first, replay, data_source="source", replace_data=False)
    assert merged[TRAIN_SPLIT].to_list() == first[TRAIN_SPLIT].to_list()
    assert merged[TEST_SPLIT].to_list() == first[TEST_SPLIT].to_list()


def test_old_duplicate_across_splits_remains_only_in_test():
    duplicate = row("same question")
    merged = merge_datasets(
        splits([duplicate], [duplicate]),
        splits(),
        data_source="source",
        replace_data=False,
    )
    assert len(merged[TRAIN_SPLIT]) == 0
    assert merged[TEST_SPLIT].to_list() == [duplicate]


def test_replace_only_changes_selected_source():
    existing = splits(
        [row("old"), row("other train", "other")], [row("other test", "other")]
    )
    merged = merge_datasets(
        existing, splits([row("new")]), data_source="source", replace_data=True
    )
    assert merged[TRAIN_SPLIT].to_list() == [row("other train", "other"), row("new")]
    assert merged[TEST_SPLIT].to_list() == [row("other test", "other")]


def test_conflicting_existing_question_requires_explicit_replacement():
    original = row("Question")
    modified = {**original, "entity": "Changed"}
    with pytest.raises(ValueError, match="Conflicting records"):
        merge_datasets(
            splits([original]),
            splits([modified]),
            data_source="source",
            replace_data=False,
        )
    assert merge_datasets(
        splits([original]), splits([modified]), data_source="source", replace_data=True
    )[TRAIN_SPLIT].to_list() == [modified]


def test_dry_run_does_not_contact_hub(tmp_path, monkeypatch):

    path = tmp_path / "input.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {key: value for key, value in row(q).items() if key != "data_source"}
            )
            for q in ["q1", "q2"]
        )
    )
    monkeypatch.setattr(
        "sys.argv", ["upload_question_rubrics.py", str(path), "source", "--dry-run"]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run contacted Hugging Face")

    monkeypatch.setattr(cli, "load_remote_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)
    cli.main()


def test_replayed_cli_batch_skips_publishing(tmp_path, monkeypatch):

    existing = splits([row("q1")], [row("q2")])
    path = tmp_path / "input.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {key: value for key, value in row(q).items() if key != "data_source"}
            )
            for q in ["q1", "q2"]
        )
    )
    monkeypatch.setattr("sys.argv", ["upload_question_rubrics.py", str(path), "source"])
    monkeypatch.setattr(cli, "load_remote_dataset", lambda *args: existing)

    def forbidden(*args, **kwargs):
        pytest.fail("an unchanged replay triggered publication")

    monkeypatch.setattr(cli, "publish_dataset", forbidden)
    cli.main()


def test_local_replay_is_deduplicated_before_split(tmp_path):

    record = {key: value for key, value in row("same").items() if key != "data_source"}
    path = tmp_path / "input.jsonl"
    path.write_text((json.dumps(record) + "\n") * 2)
    dataset = load_question_rubrics(path, "source")
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="at least two records"):
        split_dataset(dataset, 0.5)


def test_existing_hub_schema_preserves_legacy_labels_but_rejects_invalid_records():

    legacy = row("legacy")
    legacy.pop("evolution_strategies")
    legacy["question_type"] = "old label"
    aligned = align_existing_split(
        Dataset.from_list([legacy]), QUESTION_RUBRIC_FEATURES
    )
    assert aligned.to_list() == [row("legacy")]
    invalid = {**row("invalid"), "doc_ids": [999]}
    with pytest.raises(ValueError, match="union"):
        align_existing_split(Dataset.from_list([invalid]), QUESTION_RUBRIC_FEATURES)


@pytest.mark.parametrize("fail_push", [False, True])
def test_publication_keeps_private_visibility_and_propagates_push_failure(
    monkeypatch, fail_push
):
    events = []
    dataset = splits([row("train")], [row("test")])
    api = SimpleNamespace(
        create_repo=lambda *args, **kwargs: events.append(("create", args, kwargs)),
        update_repo_settings=lambda *args, **kwargs: events.append(
            ("settings", args, kwargs)
        ),
    )
    monkeypatch.setattr(publisher, "HF_TOKEN", "fake-token")
    monkeypatch.setattr(publisher, "HfApi", lambda **kwargs: api)

    def push(self, *args, **kwargs):
        events.append(("push", args, kwargs))
        assert self is dataset
        assert set(self) == set(SPLIT_NAMES)
        if fail_push:
            raise RuntimeError("Hub unavailable")

    monkeypatch.setattr(DatasetDict, "push_to_hub", push)
    if fail_push:
        with pytest.raises(RuntimeError, match="Hub unavailable"):
            publisher.publish_dataset(dataset, "account/dataset", "source")
    else:
        publisher.publish_dataset(dataset, "account/dataset", "source")
    assert [event[0] for event in events] == ["create", "settings", "push"]
    assert all(
        event[1] == ("account/dataset",) and event[2]["private"] is True
        for event in events
    )
    assert events[0][2]["repo_type"] == "dataset"
    assert events[0][2]["exist_ok"] is True
    assert events[2][2]["token"] == "fake-token"
    assert events[2][2]["commit_message"] == "Update question rubrics: source"
