#!/usr/bin/env python3
"""Train REINFORCE agent on LLM reasoning strategy selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_reasoning_optimizer.utils.config import load_config, get_project_root
from src.rl_reasoning_optimizer.utils.seeding import set_seed
from src.rl_reasoning_optimizer.utils.logging import ExperimentLogger
from src.rl_reasoning_optimizer.backends.local_backend import LocalBackend
from src.rl_reasoning_optimizer.backends.api_backend import APIBackend
from src.rl_reasoning_optimizer.env import LLMReasoningEnv
from src.rl_reasoning_optimizer.agents import ReinforceAgent
from src.rl_reasoning_optimizer.agents.policy_network import (
    build_tfidf_vectorizer,
    question_to_features,
)
from src.rl_reasoning_optimizer.eval import compute_metrics, evaluate_agent


def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    root = get_project_root()
    print("Loading config and data...", flush=True)
    cfg = load_config("default")
    set_seed(cfg.get("seed", 42))

    data_path = cfg.get("data_path", root / "data" / "math_small.jsonl")
    if isinstance(data_path, str) and not Path(data_path).is_absolute():
        data_path = root / data_path
    data = load_data(str(data_path))
    if not data:
        print("No data found. Exiting.")
        return

    train_frac = cfg.get("train_fraction", 0.7)
    val_frac = cfg.get("val_fraction", 0.15)
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))
    max_ep = cfg.get("max_episodes")
    if max_ep is not None:
        train_idx = train_idx[: int(max_ep)]
        print(f"  Limiting to max_episodes={max_ep}", flush=True)
    print(f"  Loaded {n} examples. Train: {n_train} | Val: {n_val} | Test: {n_test}", flush=True)

    backend_type = cfg.get("backend", "local")
    if backend_type == "api":
        backend = APIBackend(model=cfg.get("model", "gpt-4o-mini"))
        print("  Backend: API", flush=True)
    else:
        backend = LocalBackend(
            model=cfg.get("model", "llama2"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            use_stub_if_unavailable=cfg.get("use_stub_if_unavailable", True),
        )
        print("  Backend: local (Ollama; stub if unavailable)", flush=True)
    env = LLMReasoningEnv(
        backend=backend,
        questions=data,
        token_penalty_scale=cfg.get("token_penalty_scale", 1e-4),
        max_tokens=cfg.get("max_tokens", 1024),
    )

    questions_text = [data[i]["question"] for i in train_idx]
    vectorizer = build_tfidf_vectorizer(
        questions_text,
        max_features=cfg.get("tfidf_max_features", 500),
    )
    feature_dim = len(question_to_features("", vectorizer, 0, 0))

    agent = ReinforceAgent(
        n_actions=env.n_actions,
        feature_dim=feature_dim,
        lr=cfg.get("lr", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )

    runs_path = cfg.get("runs_path", root / "runs")
    if isinstance(runs_path, str) and not Path(runs_path).is_absolute():
        runs_path = root / runs_path
    Path(runs_path).mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        str(runs_path),
        experiment_name=cfg.get("experiment_name", "reinforce"),
        use_tensorboard=True,
        save_episode_outputs=cfg.get("save_episode_outputs", False),
    )

    n_episodes = len(train_idx)
    log_interval = cfg.get("log_interval", 10)
    token_scale = cfg.get("token_penalty_scale", 1e-4)
    print(f"\nTraining for {n_episodes} episodes (progress every {log_interval})...\n", flush=True)

    for episode in range(n_episodes):
        idx = train_idx[episode]
        state = env.reset(idx)
        features = question_to_features(
            state.question,
            vectorizer,
            state.length,
            state.num_count,
        )
        action = agent.select_action(features, deterministic=False)
        step = env.step(state, action)
        agent.store_reward(step.reward)
        agent.finish_episode()

        logger.log_episode(
            episode=episode,
            question_id=state.question_id,
            chosen_strategy=step.info.get("strategy", ""),
            llm_output=step.info.get("llm_output", ""),
            tokens_used=step.info.get("tokens_used", 0),
            reward=step.reward,
            correct=step.info.get("correct", False),
        )

        if (episode + 1) % log_interval == 0:
            mean_r = step.reward  # single step
            logger.log_metrics(episode + 1, train_reward=mean_r)
            print(f"  Episode {episode + 1}/{n_episodes} | reward={step.reward:.4f} | correct={step.info.get('correct', False)}", flush=True)

    # Validation
    print("\nRunning validation...", flush=True)
    def get_feat(s):
        return question_to_features(s.question, vectorizer, s.length, s.num_count)

    def select(features):
        return agent.select_action(features, deterministic=True)

    val_results = evaluate_agent(env, get_feat, select, val_idx)
    val_metrics = compute_metrics(val_results, token_penalty_scale=token_scale)
    logger.log_metrics(n_episodes, val_accuracy=val_metrics["accuracy"], val_avg_tokens=val_metrics["avg_tokens"])
    logger.close()
    print("\nDone. Val accuracy: {:.4f} | Val avg tokens: {:.1f}".format(val_metrics["accuracy"], val_metrics["avg_tokens"]), flush=True)


if __name__ == "__main__":
    main()
