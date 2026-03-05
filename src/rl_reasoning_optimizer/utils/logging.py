"""Logging utilities: TensorBoard and CSV to runs/."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    SummaryWriter = None  # type: ignore[misc, assignment]


class ExperimentLogger:
    """Log metrics to CSV and optionally TensorBoard under runs/."""

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str = "exp",
        use_tensorboard: bool = True,
        save_episode_outputs: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.run_dir = self._make_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self._save_episode_outputs = save_episode_outputs
        if use_tensorboard and HAS_TB and SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(self.run_dir))
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file: Any = None
        self._csv_writer: Any = None

    def _make_run_dir(self) -> Path:
        """Create a unique run subdirectory."""
        base = self.log_dir / self.experiment_name
        base.mkdir(parents=True, exist_ok=True)
        existing = list(base.iterdir()) if base.exists() else []
        run_dirs = [d for d in existing if d.is_dir() and d.name.isdigit()]
        next_num = max([int(d.name) for d in run_dirs], default=0) + 1
        return base / str(next_num)

    def log_metrics(self, step: int, **kwargs: float | int) -> None:
        """Log scalar metrics to CSV and TensorBoard."""
        row = {"step": step, **kwargs}
        if self._writer is not None:
            for k, v in row.items():
                if k != "step" and isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step)
        if self._csv_path:
            file_exists = self._csv_path.exists()
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    def log_episode(
        self,
        episode: int,
        question_id: str,
        chosen_strategy: str,
        llm_output: str,
        tokens_used: int,
        reward: float,
        correct: bool,
    ) -> None:
        """Log one episode (question) for debugging/analysis."""
        ep_path = self.run_dir / "episodes.csv"
        row = {
            "episode": episode,
            "question_id": question_id,
            "chosen_strategy": chosen_strategy,
            "tokens_used": tokens_used,
            "reward": reward,
            "correct": correct,
        }
        file_exists = ep_path.exists()
        with open(ep_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        if self._save_episode_outputs:
            out_path = self.run_dir / "outputs" / f"ep_{episode}_{question_id}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"strategy: {chosen_strategy}\ntokens: {tokens_used}\nreward: {reward}\n\n{llm_output}")

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
