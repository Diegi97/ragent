from __future__ import annotations

import datasets

from .dataset_cleaning import compute_latest_version_map, is_latest_label



def compute_latest_version_map_from_dataset(dataset, language_column = "language"):
    """
    Compute max version per base directly from a Hugging Face Dataset's language column.
    """
    if hasattr(dataset, "unique"):
        labels = dataset.unique(language_column) # type: ignore[attr-defined]
    else:
        labels = set(dataset[language_column])
    return compute_latest_version_map(labels)


def keep_latest_versions(dataset, language_column = "language"):
    """
    Filter a Hugging Face Dataset to keep rows whose `language_column` is either non-versioned
    or has the maximum numeric version for its base.
    """
    base_to_max = compute_latest_version_map_from_dataset(dataset, language_column=language_column)

    def _predicate(example):
        return is_latest_label(str(example[language_column]), base_to_max)

    return dataset.filter(_predicate)


def add_incrementing_id(dataset, id_column = "id", id_start = 0):
    """
    Add an incrementing integer id column to a Hugging Face Dataset.
    """
    ids = list(range(id_start, id_start + len(dataset)))
    return dataset.add_column(id_column, ids)


def select_columns(dataset, columns_to_keep):
    """
    Keep only the specified columns in a Hugging Face Dataset.
    """
    if hasattr(dataset, "select_columns"):
        return dataset.select_columns(list(columns_to_keep)) # type: ignore[attr-defined]
    to_drop = [c for c in dataset.column_names if c not in columns_to_keep]
    return dataset.remove_columns(to_drop)


def apply_transforms(dataset, transforms):
    """
    Apply a sequence of transforms to a Hugging Face Dataset.
    Each transform is a callable: Dataset -> Dataset.
    """
    for transform in transforms:
        dataset = transform(dataset)
    return dataset


def prepare_devdocs_dataset(
    dataset_name = "nampdn-ai/devdocs.io",
    split = "train",
    language_column = "language",
    id_column = "id",
    id_start = 0,
    keep_columns = None,
    extra_transforms = None,
):
    """
    Load a Hugging Face dataset split, then build a modular pipeline:
    - keep only latest version per base for `language_column`
    - add incrementing id column starting from `id_start`
    - optionally keep only specified columns
    - optionally apply additional transforms supplied by the caller
    """
    ds = datasets.load_dataset(dataset_name)[split]

    transforms = [
        lambda d: keep_latest_versions(d, language_column=language_column),
        lambda d: add_incrementing_id(d, id_column=id_column, id_start=id_start),
    ]

    if keep_columns is not None:
        keep_set = list(dict.fromkeys([id_column] + [c for c in keep_columns if c != id_column]))
        transforms.append(lambda d: select_columns(d, keep_set))

    if extra_transforms:
        transforms.extend(extra_transforms)

    return apply_transforms(ds, transforms)


__all__ = [
    "compute_latest_version_map_from_dataset",
    "keep_latest_versions",
    "add_incrementing_id",
    "select_columns",
    "apply_transforms",
    "prepare_devdocs_dataset",
]
