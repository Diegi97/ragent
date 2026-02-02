"""
Prepare a curriculum-ordered QA dataset and optionally upload to HuggingFace Hub.

This script takes a QA dataset directory (containing data.json and metadata.json) from the output of merge_qa_datasets.py,
applies stratified train/eval splitting, curriculum reordering based on difficulty buckets,
and optionally pushes the result to HuggingFace Hub.

Curriculum strategy:
  1. Divide samples into 3 pools by difficulty (1-3 easy, 4-6 medium, 7-9 hard)
  2. Create 3 buckets with mixed difficulties (80% primary, 10% each secondary)
  3. Shuffle within each bucket
  4. Concatenate: easy → medium → hard
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo


README_TEMPLATE = """---
pretty_name: ragent_qa_pairs
language:
- en
task_categories:
- question-answering
- agentic-search
---

# ragent_qa_pairs

This dataset contains QA pairs with `depth`, `breadth`, and `difficulty` mapped from (depth, breadth) pairs in **[1..9]**.

The `train` split is **curriculum-reordered** using difficulty-based buckets:
- Easy bucket (first ~1/3): 80% easy (diff 1-3), 10% medium, 10% hard
- Medium bucket (middle ~1/3): 80% medium (diff 4-6), 10% easy, 10% hard
- Hard bucket (last ~1/3): 80% hard (diff 7-9), 10% easy, 10% medium

Each bucket is shuffled internally, then concatenated in order: easy → medium → hard.

## Files

- `metadata.json` is uploaded alongside the dataset for full provenance.
"""

# Difficulty mapping from (depth, breadth) to difficulty level 1-9
DIFFICULTY_MAP = {
    (1, 1): 1,
    (1, 2): 2,
    (1, 3): 4,
    (2, 1): 3,
    (2, 2): 5,
    (2, 3): 7,
    (3, 1): 6,
    (3, 2): 8,
    (3, 3): 9,
}


# --------------------------
# Curriculum Reordering
# --------------------------


def curriculum_reorder_buckets(
    train: Dataset,
    seed: int = 0,
    primary_ratio: float = 0.80,
    print_stats: bool = True,
) -> Dataset:
    """
    Curriculum learning with difficulty-based buckets and mixing.

    1. Divide all samples into 3 pools by difficulty:
       - Easy pool: difficulty 1-3
       - Medium pool: difficulty 4-6
       - Hard pool: difficulty 7-9

    2. Create 3 buckets with mixed difficulties:
       - Easy bucket: 80% easy, 10% medium, 10% hard
       - Medium bucket: 80% medium, 10% easy, 10% hard
       - Hard bucket: 80% hard, 10% easy, 10% medium

    3. Randomize order within each bucket

    4. Final order: easy bucket → medium bucket → hard bucket

    Args:
        train: Dataset with a "difficulty" column (values 1-9)
        seed: Random seed for reproducibility
        primary_ratio: Fraction of bucket from primary difficulty (default 0.80)
        print_stats: Whether to print statistics

    Returns:
        Reordered Dataset
    """
    rng = np.random.default_rng(seed)

    diffs = np.asarray(train["difficulty"], dtype=np.int64)
    N = len(diffs)

    # Step 1: Divide samples into pools by actual difficulty ranges
    easy_pool = [i for i in range(N) if 1 <= diffs[i] <= 3]
    medium_pool = [i for i in range(N) if 4 <= diffs[i] <= 6]
    hard_pool = [i for i in range(N) if 7 <= diffs[i] <= 9]

    # Shuffle pools for random sampling
    rng.shuffle(easy_pool)
    rng.shuffle(medium_pool)
    rng.shuffle(hard_pool)

    # Convert to lists we can pop from
    easy_pool = list(easy_pool)
    medium_pool = list(medium_pool)
    hard_pool = list(hard_pool)

    if print_stats:
        print(
            f"Pool sizes: easy={len(easy_pool)}, medium={len(medium_pool)}, hard={len(hard_pool)}"
        )

    # Calculate bucket sizes (each bucket gets ~1/3 of total)
    bucket_size = N // 3
    remainder = N % 3

    # Distribute remainder: easy gets extra first, then medium, then hard
    easy_bucket_size = bucket_size + (1 if remainder >= 1 else 0)
    medium_bucket_size = bucket_size + (1 if remainder >= 2 else 0)
    hard_bucket_size = bucket_size

    secondary_ratio = (1.0 - primary_ratio) / 2

    def build_bucket(
        bucket_size: int,
        primary_pool: List[int],
        secondary_pool_1: List[int],
        secondary_pool_2: List[int],
        bucket_name: str,
    ) -> List[int]:
        """Build a bucket with mixed difficulties."""
        bucket: List[int] = []

        # Calculate target counts
        primary_target = int(round(bucket_size * primary_ratio))
        secondary_1_target = int(round(bucket_size * secondary_ratio))
        secondary_2_target = bucket_size - primary_target - secondary_1_target

        # Take from primary pool
        primary_take = min(primary_target, len(primary_pool))
        bucket.extend(primary_pool[:primary_take])
        del primary_pool[:primary_take]

        # Take from secondary pool 1
        secondary_1_take = min(secondary_1_target, len(secondary_pool_1))
        bucket.extend(secondary_pool_1[:secondary_1_take])
        del secondary_pool_1[:secondary_1_take]

        # Take from secondary pool 2
        secondary_2_take = min(secondary_2_target, len(secondary_pool_2))
        bucket.extend(secondary_pool_2[:secondary_2_take])
        del secondary_pool_2[:secondary_2_take]

        # If we couldn't fill the bucket (pool exhaustion), fill from remaining pools
        shortfall = bucket_size - len(bucket)
        if shortfall > 0:
            for pool in [primary_pool, secondary_pool_1, secondary_pool_2]:
                take = min(shortfall, len(pool))
                if take > 0:
                    bucket.extend(pool[:take])
                    del pool[:take]
                    shortfall -= take
                if shortfall <= 0:
                    break

        if print_stats:
            bucket_diffs = [diffs[i] for i in bucket]
            easy_count = sum(1 for d in bucket_diffs if 1 <= d <= 3)
            medium_count = sum(1 for d in bucket_diffs if 4 <= d <= 6)
            hard_count = sum(1 for d in bucket_diffs if 7 <= d <= 9)
            print(
                f"{bucket_name} bucket: size={len(bucket)}, "
                f"easy={easy_count} ({100 * easy_count / len(bucket):.1f}%), "
                f"medium={medium_count} ({100 * medium_count / len(bucket):.1f}%), "
                f"hard={hard_count} ({100 * hard_count / len(bucket):.1f}%)"
            )

        return bucket

    # Step 2: Build buckets with mixing
    easy_bucket = build_bucket(
        easy_bucket_size, easy_pool, medium_pool, hard_pool, "Easy"
    )
    medium_bucket = build_bucket(
        medium_bucket_size, medium_pool, easy_pool, hard_pool, "Medium"
    )
    hard_bucket = build_bucket(
        hard_bucket_size, hard_pool, easy_pool, medium_pool, "Hard"
    )

    # Step 3: Shuffle within each bucket
    rng.shuffle(easy_bucket)
    rng.shuffle(medium_bucket)
    rng.shuffle(hard_bucket)

    # Step 4: Concatenate: easy → medium → hard
    out = easy_bucket + medium_bucket + hard_bucket

    # Handle any leftover samples (safety net)
    leftovers = easy_pool + medium_pool + hard_pool
    if leftovers:
        rng.shuffle(leftovers)
        out.extend(leftovers)

    # Sanity check
    if len(out) != N or len(set(out)) != N:
        raise RuntimeError(
            f"Reordering failed to produce a valid permutation. "
            f"Got {len(out)} items with {len(set(out))} unique, expected {N}."
        )

    reordered = train.select(out)

    if print_stats:
        dd = np.asarray(reordered["difficulty"], dtype=np.float64)
        third = N // 3
        a = dd[:third].mean() if N >= 3 else dd.mean()
        b = dd[third : 2 * third].mean() if N >= 3 else dd.mean()
        c = dd[2 * third :].mean() if N >= 3 else dd.mean()
        print(f"Mean difficulty by thirds: early={a:.3f}, mid={b:.3f}, late={c:.3f}")

    return reordered


# --------------------------
# Stratified Split
# --------------------------


def stratified_split_indices(
    data_source: List[str],
    difficulty: List[int],
    eval_ratio: float = 0.02,
    seed: int = 0,
    min_eval_per_stratum: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Stratify into train/eval by (data_source, difficulty).

    Policy:
      - For each stratum, allocate floor(n * eval_ratio), but at least min_eval_per_stratum
        if the stratum has enough items (>= 2).
      - If a stratum has only 1 item, it stays in train.
    """
    if not (0.0 < eval_ratio < 1.0):
        raise ValueError("eval_ratio must be in (0,1)")

    rng = np.random.default_rng(seed)
    N = len(difficulty)
    if len(data_source) != N:
        raise ValueError("data_source and difficulty must have same length")

    # Group indices by stratum
    strata: Dict[str, List[int]] = {}
    for i, (src, d) in enumerate(zip(data_source, difficulty)):
        key = f"{src}__d{int(d)}"
        strata.setdefault(key, []).append(i)

    train_idx: List[int] = []
    eval_idx: List[int] = []

    for key, idxs in strata.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)

        if n < 2:
            train_idx.extend(idxs)
            continue

        k = int(np.floor(n * eval_ratio))
        k = max(k, min_eval_per_stratum)
        k = min(k, n - 1)  # never take all

        eval_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    rng.shuffle(train_idx)
    rng.shuffle(eval_idx)

    return train_idx, eval_idx


# --------------------------
# Main Pipeline
# --------------------------


def prepare_curriculum_dataset(
    folder_path: str,
    repo_id: str | None = None,
    private: bool = True,
    eval_ratio: float = 0.10,
    seed: int = 0,
    primary_ratio: float = 0.80,
    upload: bool = False,
) -> DatasetDict:
    """
    Load merged QA data, apply stratified split and curriculum reorder, optionally upload.

    Args:
        folder_path: Path to directory with data.json and metadata.json
        repo_id: HuggingFace repo ID (required if upload=True)
        private: Whether the HF repo should be private
        eval_ratio: Fraction of data for eval split
        seed: Random seed
        primary_ratio: Fraction of bucket from primary difficulty (default 0.80)
        upload: Whether to upload to HuggingFace Hub

    Returns:
        DatasetDict with train and eval splits
    """
    data_path = os.path.join(folder_path, "data.json")
    meta_path = os.path.join(folder_path, "metadata.json")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Missing data.json at: {data_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata.json at: {meta_path}")

    print(f"Loading data from: {folder_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(records)} records")

    # Normalize + compute difficulty field
    normalized: List[Dict[str, Any]] = []
    for r in records:
        r = dict(r)  # shallow copy

        # Rename dataset_id -> data_source
        if "dataset_id" in r:
            r["data_source"] = r.pop("dataset_id")

        # Compute difficulty from depth/breadth
        info = r.get("info", {}) or {}
        depth = int(info.get("depth", 1))
        breadth = int(info.get("breadth", 1))
        r["difficulty"] = DIFFICULTY_MAP.get((depth, breadth), 5)

        normalized.append(r)

    ds = Dataset.from_list(normalized)

    # Stratified split
    train_idx, eval_idx = stratified_split_indices(
        data_source=ds["data_source"],
        difficulty=ds["difficulty"],
        eval_ratio=eval_ratio,
        seed=seed,
        min_eval_per_stratum=1,
    )

    train = ds.select(train_idx)
    eval_ = ds.select(eval_idx)

    print(f"\nSplit sizes: train={len(train)}, eval={len(eval_)}")

    # Curriculum reorder train
    print("\nApplying curriculum reorder...")
    train = curriculum_reorder_buckets(
        train=train,
        seed=seed,
        primary_ratio=primary_ratio,
    )

    # Show sample difficulties
    train_difficulties = train["difficulty"]
    print(f"\nFirst 20 train difficulties: {train_difficulties[:20]}")
    print(f"Last 20 train difficulties: {train_difficulties[-20:]}")

    # Transform to verifiers format:
    # - prompt: renamed from question
    # - answer: kept as-is
    # - info: JSON string with data_source, difficulty, doc_indices, and original info fields
    def to_verifiers_format(example: Dict[str, Any]) -> Dict[str, Any]:
        info = example.get("info", {}) or {}
        new_info = {
            "data_source": example.get("data_source", "unknown"),
            "difficulty": example.get("difficulty", 5),
            "doc_indices": example.get("doc_indices", []),
            **info,  # Include all original info fields (depth, breadth, concept, etc.)
        }
        return {
            "id": example.get("id"),
            "prompt": example.get("question", ""),
            "answer": example.get("answer", ""),
            "info": json.dumps(new_info),
        }

    train = train.map(to_verifiers_format, remove_columns=train.column_names)
    eval_ = eval_.map(to_verifiers_format, remove_columns=eval_.column_names)

    dsd = DatasetDict({"train": train, "eval": eval_})

    # Upload to HuggingFace Hub if requested
    if upload:
        if not repo_id:
            raise ValueError("repo_id is required when upload=True")

        print(f"\nUploading to HuggingFace Hub: {repo_id}")

        api = HfApi()

        create_repo(
            repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset"
        )
        dsd.push_to_hub(repo_id, private=private)

        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add metadata.json",
        )

        api.upload_file(
            path_or_fileobj=README_TEMPLATE.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

        print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")

    return dsd


if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "data/merged/qa-merged-20260130-142041-2603pairs"
    REPO_ID = "diegi97/ragent_qa_pairs"
    EVAL_RATIO = 0.10
    SEED = 123
    PRIMARY_RATIO = 0.80  # 80% primary difficulty, 10% each secondary
    UPLOAD = True  # Set to True to push to HuggingFace Hub

    print("=" * 60)
    print("Curriculum Dataset Preparation")
    print("=" * 60)
    print(f"Input folder: {FOLDER_PATH}")
    print(f"Repo ID: {REPO_ID}")
    print(f"Eval ratio: {EVAL_RATIO}")
    print(f"Seed: {SEED}")
    print(f"Primary ratio: {PRIMARY_RATIO}")
    print(f"Upload: {UPLOAD}")
    print("=" * 60)

    dsd = prepare_curriculum_dataset(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        private=True,
        eval_ratio=EVAL_RATIO,
        seed=SEED,
        primary_ratio=PRIMARY_RATIO,
        upload=UPLOAD,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Train samples: {len(dsd['train'])}")
    print(f"Eval samples: {len(dsd['eval'])}")
    print("=" * 60)
