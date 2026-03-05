#!/usr/bin/env python3
"""Run epsilon-greedy bandit baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_reasoning_optimizer.utils.config import load_config, get_project_root
from src.rl_reasoning_optimizer.utils.seeding import set_seed
from src.rl_reasoning_optimizer.backends.local_backend import LocalBackend
from src.rl_reasoning_optimizer.backends.api_backend import APIBackend
from src.rl_reasoning_optimizer.env import LLMReasoningEnv
from src.rl_reasoning_optimizer.agents import EpsilonGreedyBandit
from src.rl_reasoning_optimizer.agents.policy_network import question_to_features, build_tfidf_vectorizer
from src.rl_reasoning_optimizer.eval import evaluate_agent, compute_metrics


def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    root = get_project_root()
    cfg = load_config("default")
    set_seed(cfg.get("seed", 42))

    data_path = cfg.get("data_path", root / "data" / "math_small.jsonl")
    if isinstance(data_path, str) and not Path(data_path).is_absolute():
        data_path = root / data_path
    data = load_data(str(data_path))
    if not data:
        print("No data. Exiting.")
        return

    n_train = int(len(data) * cfg.get("train_fraction", 0.7))
    indices = list(range(n_train))
    max_ep = cfg.get("max_episodes")
    if max_ep is not None:
        indices = indices[: int(max_ep)]
    n_run = len(indices)
    print("Bandit baseline: {} episodes".format(n_run), flush=True)

    backend_type = cfg.get("backend", "local")
    if backend_type == "api":
        backend = APIBackend(model=cfg.get("model", "gpt-4o-mini"))
    else:
        backend = LocalBackend(
            model=cfg.get("model", "llama2"),
            use_stub_if_unavailable=cfg.get("use_stub_if_unavailable", True),
        )

    env = LLMReasoningEnv(
        backend=backend,
        questions=data,
        token_penalty_scale=cfg.get("token_penalty_scale", 1e-4),
        max_tokens=cfg.get("max_tokens", 1024),
    )

    questions_text = [data[i]["question"] for i in indices]
    vectorizer = build_tfidf_vectorizer(questions_text, max_features=cfg.get("tfidf_max_features", 500))

    bandit = EpsilonGreedyBandit(n_actions=env.n_actions, epsilon=cfg.get("bandit_epsilon", 0.1), seed=cfg.get("seed"))

    def get_feat(s):
        return question_to_features(s.question, vectorizer, s.length, s.num_count)

    def select(features):
        return bandit.select_action(features, deterministic=False)

    log_interval = max(1, n_run // 10)
    results = []
    for i, idx in enumerate(indices):
        state = env.reset(idx)
        features = get_feat(state)
        action = select(features)
        step = env.step(state, action)
        bandit.update(action, step.reward)
        results.append({
            "correct": step.info.get("correct", False),
            "reward": step.reward,
            "tokens_used": step.info.get("tokens_used", 0),
        })
        if (i + 1) % log_interval == 0:
            print("  Episode {}/{}".format(i + 1, n_run), flush=True)

    metrics = compute_metrics(results, token_penalty_scale=cfg.get("token_penalty_scale", 1e-4))
    print("Done. Accuracy: {:.4f} | Avg tokens: {:.1f} | Mean reward: {:.4f}".format(metrics["accuracy"], metrics["avg_tokens"], metrics["mean_reward"]), flush=True)


if __name__ == "__main__":
    main()
