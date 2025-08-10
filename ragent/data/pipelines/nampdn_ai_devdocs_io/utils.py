import re

_LEADING_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)")

def extract_base_and_version(label):
    if "~" not in label:
        return label, None, ""
    base, suffix = label.split("~", 1)
    match = _LEADING_VERSION_RE.match(suffix)
    if not match:
        return base, None, suffix
    version_str = match.group(1)
    version_tuple = tuple(int(p) for p in version_str.split("."))
    return base, version_tuple, suffix

def is_latest_label(label, base_to_max_version):
    base, version_tuple, _ = extract_base_and_version(label)
    if version_tuple is None:
        return True
    return base_to_max_version.get(base) == version_tuple

def compute_latest_version_map(labels):
    base_to_max_version = {}
    for label in labels:
        base, version_tuple, _ = extract_base_and_version(label)
        if version_tuple is None:
            continue
        current = base_to_max_version.get(base)
        if current is None or version_tuple > current:
            base_to_max_version[base] = version_tuple
    return base_to_max_version

def compute_latest_version_map_from_dataset(dataset, language_column = "language"):
    if hasattr(dataset, "unique"):
        labels = dataset.unique(language_column) # type: ignore[attr-defined]
    else:
        labels = set(dataset[language_column])
    return compute_latest_version_map(labels)
