from __future__ import annotations

import re

_LEADING_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def extract_base_and_version(label):
    """
    Split a label like "dart~2", "qt~5.15", or "openjdk~8_gui" into:
    - base (left of '~')
    - version tuple if a leading numeric version exists on the right of '~'
    - full suffix (right of '~', unchanged)

    If no '~' exists or no leading numeric sequence is present, version is None.
    """
    if "~" not in label:
        return label, None, ""
    base, suffix = label.split("~", 1)
    match = _LEADING_VERSION_RE.match(suffix)
    if not match:
        return base, None, suffix
    version_str = match.group(1)
    version_tuple = tuple(int(p) for p in version_str.split("."))
    return base, version_tuple, suffix


def compute_latest_version_map(labels):
    """
    For all labels, compute the maximum version per base among versioned labels.
    Non-versioned labels are ignored in the map.
    """
    base_to_max_version = {}
    for label in labels:
        base, version_tuple, _ = extract_base_and_version(label)
        if version_tuple is None:
            continue
        current = base_to_max_version.get(base)
        if current is None or version_tuple > current:
            base_to_max_version[base] = version_tuple
    return base_to_max_version


def is_latest_label(label, base_to_max_version):
    """
    Decide if a label should be kept under the "keep latest version per base" rule.
    - Non-versioned labels are kept.
    - Versioned labels are kept only if their version equals the maximum for their base.
    """
    base, version_tuple, _ = extract_base_and_version(label)
    if version_tuple is None:
        return True
    return base_to_max_version.get(base) == version_tuple


def filter_latest_labels(labels):
    """
    Return the subset of labels to keep under the rule:
    - Keep non-versioned labels
    - Keep labels whose version equals the maximum version for their base
    """
    base_to_max = compute_latest_version_map(labels)
    return [label for label in labels if is_latest_label(label, base_to_max)]


def clean_dataframe_latest_versions(df, language_column = "language"):
    """
    Return a filtered copy of the DataFrame keeping only rows where `language_column`
    is either non-versioned or has the maximum numeric version among rows sharing the same base.

    Import of pandas occurs inside the function to avoid hard dependency when this utility is imported.
    """
    if language_column not in df.columns:
        raise KeyError(f"Column '{language_column}' not found in DataFrame")

    labels = df[language_column].astype(str).tolist()
    base_to_max = compute_latest_version_map(labels)
    mask = df[language_column].map(lambda s: is_latest_label(str(s), base_to_max))
    return df[mask].reset_index(drop=True)


__all__ = [
    "extract_base_and_version",
    "compute_latest_version_map",
    "is_latest_label",
    "filter_latest_labels",
    "clean_dataframe_latest_versions",
]


