"""NASA's public Lessons Learned Information System (LLIS)."""

from pathlib import Path

from datasets import Dataset, load_from_disk

from ragent_core.data_sources import DataSourceSpec

DATA_DIR = Path("data/nasa_lessons_learned")
DESCRIPTION = (
    "NASA's public Lessons Learned Information System (LLIS), covering engineering "
    "and operational lessons from NASA programs and projects. Each document is one "
    "lesson, with its original NASA lesson number, source URL, organizational "
    "context, abstract, driving event, lessons learned, and recommendations."
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
        dataset=dataset, name="nasa_lessons_learned", description=DESCRIPTION
    )
