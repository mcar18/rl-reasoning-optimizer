#!/usr/bin/env python3
"""Evaluate trained model and baselines with bootstrap CIs."""

from __future__ import annotations

import argparse
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
from src.rl_reasoning_optimizer.agents import ReinforceAgent, EpsilonGreedyBandit
from src.rl_reasoning_optimizer.agents.policy_network import (
    build_tfidf_vectorizer,
    question_to_features,
)
from src.rl_reasoning_optimizer.eval import (
    evaluate_agent,
    compute_metrics,
    bootstrap_ci,
    run_random_baseline,
    run_best_fixed_baseline,
    find_best_fixed_strategy,
)


def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model and baselines with bootstrap CIs.")
    parser.add_argument("--config", default="default", help="Config name (default, local_ollama, api_model)")
    args = parser.parse_args()
    root = get_project_root()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_path = cfg.get("data_path", root / "data" / "math_small.jsonl")
    if isinstance(data_path, str) and not Path(data_path).is_absolute():
        data_path = root / data_path
    data = load_data(str(data_path))
    if not data:
        print("No data.")
        return

    n = len(data)
    n_train = int(n * cfg.get("train_fraction", 0.7))
    n_val = int(n * cfg.get("val_fraction", 0.15))
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))
    if not test_idx:
        test_idx = val_idx

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

    questions_text = [data[i]["question"] for i in train_idx]
    vectorizer = build_tfidf_vectorizer(questions_text, max_features=cfg.get("tfidf_max_features", 500))
    token_scale = cfg.get("token_penalty_scale", 1e-4)
    n_boot = cfg.get("bootstrap_n", 200)
    conf = cfg.get("confidence", 0.95)
    best_fixed_max = cfg.get("best_fixed_max_questions", 50)

    def get_feat(s):
        return question_to_features(s.question, vectorizer, s.length, s.num_count)

    def accuracy_fn(results):
        return compute_metrics(results, token_penalty_scale=token_scale)["accuracy"]

    print("Random baseline...", flush=True)
    random_results = run_random_baseline(env, get_feat, test_idx, seed=cfg.get("seed"))
    r_metrics = compute_metrics(random_results, token_penalty_scale=token_scale)
    r_acc, r_lo, r_hi = bootstrap_ci(random_results, accuracy_fn, n_bootstrap=n_boot, confidence=conf, seed=cfg.get("seed"))
    print("Random baseline: accuracy = {:.4f} [{:.4f}, {:.4f}], avg_tokens = {:.1f}".format(r_acc, r_lo, r_hi, r_metrics["avg_tokens"]))

    print("Finding best fixed strategy (max {} train questions)...".format(best_fixed_max), flush=True)
    best_idx = find_best_fixed_strategy(env, get_feat, train_idx, token_penalty_scale=token_scale, max_questions=best_fixed_max)
    fixed_results = run_best_fixed_baseline(env, get_feat, test_idx, best_idx)
    f_metrics = compute_metrics(fixed_results, token_penalty_scale=token_scale)
    f_acc, f_lo, f_hi = bootstrap_ci(fixed_results, accuracy_fn, n_bootstrap=n_boot, confidence=conf, seed=cfg.get("seed"))
    print("Best fixed (strategy {}): accuracy = {:.4f} [{:.4f}, {:.4f}], avg_tokens = {:.1f}".format(best_idx, f_acc, f_lo, f_hi, f_metrics["avg_tokens"]))

    print("Bandit on test...", flush=True)
    bandit = EpsilonGreedyBandit(env.n_actions, epsilon=0.1, seed=cfg.get("seed"))
    bandit_results = []
    for idx in test_idx:
        state = env.reset(idx)
        features = get_feat(state)
        action = bandit.select_action(features, deterministic=False)
        step = env.step(state, action)
        bandit.update(action, step.reward)
        bandit_results.append({"correct": step.info.get("correct", False), "reward": step.reward, "tokens_used": step.info.get("tokens_used", 0)})
    b_metrics = compute_metrics(bandit_results, token_penalty_scale=token_scale)
    b_acc, b_lo, b_hi = bootstrap_ci(bandit_results, accuracy_fn, n_bootstrap=n_boot, confidence=conf, seed=cfg.get("seed"))
    print("Bandit: accuracy = {:.4f} [{:.4f}, {:.4f}], avg_tokens = {:.1f}".format(b_acc, b_lo, b_hi, b_metrics["avg_tokens"]))

    # REINFORCE: load trained policy if checkpoint exists, else use untrained
    runs_path = root / cfg.get("runs_path", "runs")
    if isinstance(runs_path, str):
        runs_path = root / runs_path
    exp_dir = runs_path / cfg.get("experiment_name", "reinforce")
    reinf_metrics = None
    reinf_ci = (0.0, 0.0)
    if exp_dir.exists():
        run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name), reverse=True)
        for run_dir in run_dirs:
            policy_path = run_dir / "policy.pt"
            meta_path = run_dir / "checkpoint_meta.json"
            if policy_path.exists() and meta_path.exists():
                print("REINFORCE (trained)...", flush=True)
                import torch
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                agent = ReinforceAgent(n_actions=meta["n_actions"], feature_dim=meta["feature_dim"])
                agent.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
                reinf_results = evaluate_agent(
                    env,
                    get_feat,
                    lambda features: agent.select_action(features, deterministic=True),
                    test_idx,
                )
                reinf_metrics = compute_metrics(reinf_results, token_penalty_scale=token_scale)
                r_acc_reinf, r_lo_reinf, r_hi_reinf = bootstrap_ci(reinf_results, accuracy_fn, n_bootstrap=n_boot, confidence=conf, seed=cfg.get("seed"))
                reinf_ci = (r_lo_reinf, r_hi_reinf)
                print("REINFORCE (trained): accuracy = {:.4f} [{:.4f}, {:.4f}], avg_tokens = {:.1f}".format(
                    reinf_metrics["accuracy"], r_lo_reinf, r_hi_reinf, reinf_metrics["avg_tokens"]), flush=True)
                break
    if reinf_metrics is None:
        print("REINFORCE (untrained, no checkpoint found)...", flush=True)
        feature_dim = len(question_to_features("", vectorizer, 0, 0))
        agent = ReinforceAgent(n_actions=env.n_actions, feature_dim=feature_dim)
        reinf_results = evaluate_agent(
            env,
            get_feat,
            lambda features: agent.select_action(features, deterministic=True),
            test_idx,
        )
        reinf_metrics = compute_metrics(reinf_results, token_penalty_scale=token_scale)
        reinf_ci = (reinf_metrics["accuracy"], reinf_metrics["accuracy"])
        print("REINFORCE (untrained): accuracy = {:.4f}, avg_tokens = {:.1f}".format(reinf_metrics["accuracy"], reinf_metrics["avg_tokens"]), flush=True)

    # Save results for plot_results.py to use
    results_path = root / "results" / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    eval_results = {
        "random": {"accuracy": r_acc, "ci": (r_lo, r_hi), "avg_tokens": r_metrics["avg_tokens"]},
        "best_fixed": {"accuracy": f_acc, "ci": (f_lo, f_hi), "avg_tokens": f_metrics["avg_tokens"]},
        "bandit": {"accuracy": b_acc, "ci": (b_lo, b_hi), "avg_tokens": b_metrics["avg_tokens"]},
        "reinforce": {"accuracy": reinf_metrics["accuracy"], "ci": tuple(reinf_ci), "avg_tokens": reinf_metrics["avg_tokens"]},
    }
    with open(results_path, "w", encoding="utf-8") as f:
        _json.dump(eval_results, f, indent=2)
    print("Results saved to", results_path, flush=True)


if __name__ == "__main__":
    main()
