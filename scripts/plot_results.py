#!/usr/bin/env python3
"""Generate training and evaluation plots; save to results/plots."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = ROOT / "results" / "plots"
RUNS_DIR = ROOT / "runs"


def load_metrics_csv(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = {}
            for k, v in row.items():
                if isinstance(v, list):
                    continue
                try:
                    if k == "step":
                        out[k] = int(v) if isinstance(v, (int, float)) else int(float(str(v).strip()))
                    else:
                        out[k] = float(v) if isinstance(v, (int, float)) else float(str(v).strip())
                except (ValueError, TypeError):
                    continue
            if out:
                rows.append(out)
    return rows


def plot_training_reward(run_dir: Path, out_path: Path) -> None:
    rows = load_metrics_csv(run_dir)
    if not rows:
        return
    steps = [r["step"] for r in rows]
    rewards = [r.get("train_reward", r.get("mean_reward", 0)) for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Train reward")
    plt.title("Training reward curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy_comparison(labels: list[str], accuracies: list[float], cis: list[tuple[float, float]], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    err_lo = [acc - lo for acc, (lo, hi) in zip(accuracies, cis)]
    err_hi = [hi - acc for acc, (lo, hi) in zip(accuracies, cis)]
    plt.bar(x, accuracies, yerr=np.array([err_lo, err_hi]), capsize=5)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison vs baselines")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_token_comparison(labels: list[str], avg_tokens: list[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(labels)), avg_tokens, color="steelblue", alpha=0.8)
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("Average tokens")
    plt.title("Token usage comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if RUNS_DIR.exists():
        for exp in ["reinforce"]:
            exp_dir = RUNS_DIR / exp
            if exp_dir.exists():
                run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))
                if run_dirs:
                    plot_training_reward(run_dirs[-1], OUT_DIR / "training_reward.png")

    # Use saved eval results if available; otherwise placeholders
    results_file = ROOT / "results" / "eval_results.json"
    if results_file.exists():
        import json
        with open(results_file, encoding="utf-8") as f:
            eval_results = json.load(f)
        labels = ["Random", "Best fixed", "Bandit", "REINFORCE"]
        accs = [eval_results["random"]["accuracy"], eval_results["best_fixed"]["accuracy"], eval_results["bandit"]["accuracy"], eval_results["reinforce"]["accuracy"]]
        cis = [tuple(eval_results["random"]["ci"]), tuple(eval_results["best_fixed"]["ci"]), tuple(eval_results["bandit"]["ci"]), tuple(eval_results["reinforce"]["ci"])]
        tokens = [eval_results["random"]["avg_tokens"], eval_results["best_fixed"]["avg_tokens"], eval_results["bandit"]["avg_tokens"], eval_results["reinforce"]["avg_tokens"]]
        plot_accuracy_comparison(labels, accs, cis, OUT_DIR / "accuracy_comparison.png")
        plot_token_comparison(labels, tokens, OUT_DIR / "token_usage_comparison.png")
        print("Plots use results from results/eval_results.json", flush=True)
    else:
        plot_accuracy_comparison(
            ["Random", "Best fixed", "Bandit", "REINFORCE"],
            [0.25, 0.45, 0.42, 0.48],
            [(0.20, 0.30), (0.40, 0.50), (0.37, 0.47), (0.43, 0.53)],
            OUT_DIR / "accuracy_comparison.png",
        )
        plot_token_comparison(
            ["Random", "Best fixed", "Bandit", "REINFORCE"],
            [800, 600, 650, 550],
            OUT_DIR / "token_usage_comparison.png",
        )
        print("No results/eval_results.json found; comparison plots use placeholder data. Run evaluate_models.py first.", flush=True)
    print("Plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
