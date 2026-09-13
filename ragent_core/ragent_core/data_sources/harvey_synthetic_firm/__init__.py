"""Harvey's Calderwood & Harkness synthetic law firm document corpus."""

from pathlib import Path

from datasets import Dataset, load_from_disk

from ragent_core.data_sources import DataSourceSpec

DATA_DIR = Path("data/harvey_synthetic_firm")
DESCRIPTION = (
    "The fictional law firm Calderwood & Harkness, released by Harvey as part of "
    "Legal Agent Bench. The corpus contains matter work product including legal "
    "agreements, memoranda, emails, spreadsheets, and presentations. Each document "
    "preserves its source path and matter identifier. Benchmark questions, answers, "
    "and grading rubrics are excluded. All content describes synthetic clients "
    "and matters, rather than real legal advice or real client records."
)


def load_data_source() -> DataSourceSpec:
    from .prepare import prepare_dataset

    dataset_path = DATA_DIR / "dataset"
    if not dataset_path.exists():
        prepare_dataset(DATA_DIR)
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {dataset_path}")
    return DataSourceSpec(
        dataset=dataset, name="harvey_synthetic_firm", description=DESCRIPTION
    )
