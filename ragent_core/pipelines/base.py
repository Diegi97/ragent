from __future__ import annotations

import json
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from ragent_core.types import QA

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetadata:
    """Describes high-level information about a pipeline for discovery/UI purposes."""

    name: str
    description: str
    task: str = "qa-generation"


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing behavior."""

    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 10
    resume_from: Optional[str] = None
    enable_checkpointing: bool = True


def _qa_to_dict(qa: Any, idx: int) -> dict:
    """Convert a QA object to a serializable dictionary."""
    record: dict[str, Any] = {
        "id": idx,
        "doc_ids": list(qa.doc_ids),
        "num_docs": len(qa.doc_ids),
    }

    if hasattr(qa, "questions"):
        questions = list(qa.questions)
        answers = list(qa.answers)
        record.update(
            {
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "num_answers": len(answers),
            }
        )
    else:
        record.update(
            {
                "question": qa.question,
                "answer": qa.answer,
            }
        )

    info = qa.info
    if info:
        record["info"] = info

    return record


def _slugify(value: str) -> str:
    """Convert a string to a URL-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", value.strip().lower(), "-")
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "run"


class CheckpointManager:
    """Manages checkpointing and resumption for pipeline execution."""

    def __init__(
        self,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_interval: int = 10,
        pipeline_name: str = "pipeline",
        run_identifier: Optional[str] = None,
    ) -> None:
        self._checkpoint_interval = max(1, checkpoint_interval)
        self._pipeline_name = pipeline_name
        self._run_identifier = run_identifier or self._generate_run_identifier()

        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)
        else:
            self._checkpoint_dir = (
                Path("checkpoints") / self._pipeline_name / self._run_identifier
            )

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint_count = 0
        self._checkpoint_count = 0

    def _generate_run_identifier(self) -> str:
        """Generate a unique run identifier based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        pipeline_slug = _slugify(self._pipeline_name.replace("/", "-"))
        return f"{pipeline_slug}-{timestamp}"

    @property
    def run_identifier(self) -> str:
        """Return the current run identifier."""
        return self._run_identifier

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory."""
        return self._checkpoint_dir

    def should_checkpoint(self, current_count: int) -> bool:
        """Determine if a checkpoint should be saved based on current progress."""
        if current_count <= self._last_checkpoint_count:
            return False
        return (
            current_count - self._last_checkpoint_count
        ) >= self._checkpoint_interval

    def save_checkpoint(
        self,
        qas: Sequence[QA],
        state: Optional[dict[str, Any]] = None,
        is_final: bool = False,
    ) -> Path:
        """
        Save a checkpoint with the current QA pairs and optional state.

        Args:
            qas: The list of QA pairs to checkpoint.
            state: Optional pipeline state to preserve (e.g., processed indices, concepts).
            is_final: If True, marks this as the final checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        self._checkpoint_count += 1
        self._last_checkpoint_count = len(qas)

        if is_final:
            checkpoint_filename = "checkpoint_final.json"
        else:
            checkpoint_filename = (
                f"checkpoint_{self._checkpoint_count:04d}_{len(qas):06d}.json"
            )

        checkpoint_path = self._checkpoint_dir / checkpoint_filename

        checkpoint_data = {
            "metadata": {
                "pipeline_name": self._pipeline_name,
                "run_identifier": self._run_identifier,
                "checkpoint_number": self._checkpoint_count,
                "num_qas": len(qas),
                "created_at": datetime.now().isoformat(),
                "is_final": is_final,
            },
            "qas": [_qa_to_dict(qa, i + 1) for i, qa in enumerate(qas)],
            "state": state or {},
        }

        # Write atomically by writing to temp file then renaming
        temp_path = checkpoint_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        temp_path.rename(checkpoint_path)

        logger.info(
            "Checkpoint saved: %s (%d QA pairs)",
            checkpoint_path,
            len(qas),
        )

        return checkpoint_path

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | Path,
    ) -> tuple[list[QA], dict[str, Any], dict[str, Any]]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Tuple of (list of QA objects, state dict, metadata dict).
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        with path.open("r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        metadata = checkpoint_data.get("metadata", {})
        state = checkpoint_data.get("state", {})
        raw_qas = checkpoint_data.get("qas", [])

        qas: list[QA] = []
        for item in raw_qas:
            if not isinstance(item, dict):
                continue

            # Handle both single QA and multi-step QA formats
            if "questions" in item:
                # Multi-step QA format - reconstruct as single QA with first question/answer
                questions = item.get("questions", [])
                answers = item.get("answers", [])
                question = questions[0] if questions else ""
                answer = answers[0] if answers else ""
            else:
                question = item.get("question", "")
                answer = item.get("answer", "")

            if not question or not answer:
                continue

            qa = QA(
                question=question,
                answer=answer,
                doc_ids=item.get("doc_ids", []),
                info=item.get("info", {}),
            )
            qas.append(qa)

        logger.info(
            "Loaded checkpoint: %s (%d QA pairs, checkpoint #%d)",
            path,
            len(qas),
            metadata.get("checkpoint_number", 0),
        )

        return qas, state, metadata

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
        """
        Find the latest checkpoint file in a directory.

        Args:
            checkpoint_dir: Directory to search for checkpoints.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found.
        """
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists():
            return None

        checkpoint_files = list(dir_path.glob("checkpoint_*.json"))
        if not checkpoint_files:
            return None

        # Sort by modification time, most recent first
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files[0]

    def cleanup_intermediate_checkpoints(self, keep_final: bool = True) -> int:
        """
        Remove intermediate checkpoints, optionally keeping the final one.

        Args:
            keep_final: If True, keeps the final checkpoint.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        for checkpoint_file in self._checkpoint_dir.glob("checkpoint_*.json"):
            if keep_final and checkpoint_file.name == "checkpoint_final.json":
                continue
            checkpoint_file.unlink()
            removed += 1

        if removed > 0:
            logger.info("Cleaned up %d intermediate checkpoint(s)", removed)

        return removed


class BasePipeline:
    """Abstract base for all pipelines with checkpointing support."""

    metadata: PipelineMetadata = PipelineMetadata(
        name="base",
        description="Abstract base pipeline",
    )

    def __init__(self) -> None:
        self._checkpoint_manager: Optional[CheckpointManager] = None

    def _init_checkpoint_manager(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        run_identifier: Optional[str] = None,
    ) -> CheckpointManager:
        """Initialize the checkpoint manager for this pipeline run."""
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            pipeline_name=self.metadata.name,
            run_identifier=run_identifier,
        )
        return self._checkpoint_manager

    def _maybe_checkpoint(
        self,
        qas: Sequence[QA],
        state: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Save a checkpoint if conditions are met.

        Args:
            qas: Current list of QA pairs.
            state: Optional pipeline state to preserve.
            force: If True, save regardless of interval.

        Returns:
            Path to checkpoint if saved, None otherwise.
        """
        if self._checkpoint_manager is None:
            return None

        if force or self._checkpoint_manager.should_checkpoint(len(qas)):
            return self._checkpoint_manager.save_checkpoint(qas, state)

        return None

    def _save_final_checkpoint(
        self,
        qas: Sequence[QA],
        state: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save the final checkpoint after pipeline completion."""
        if self._checkpoint_manager is None:
            return None

        return self._checkpoint_manager.save_checkpoint(qas, state, is_final=True)

    @abstractmethod
    async def generate(self, config) -> list[QA]:
        """Execute the pipeline and return generated QA pairs."""
        raise NotImplementedError

    def split_train_eval(
        self,
        qas: Sequence[QA],
        *,
        eval_size: float | int | None = None,
        train_size: float | int | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> tuple[list[QA], list[QA]]:
        """
        Split QA pairs into train/eval partitions.

        Default implementation uses random splitting. Subclasses may override
        for stratified splitting based on recipe or other criteria.
        """
        from sklearn.model_selection import train_test_split

        qas_list = list(qas)
        train_qas, eval_qas = train_test_split(
            qas_list,
            test_size=eval_size,
            train_size=train_size,
            random_state=seed,
            shuffle=shuffle,
        )
        return list(train_qas), list(eval_qas)
