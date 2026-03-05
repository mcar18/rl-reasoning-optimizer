#!/usr/bin/env python3
"""Check that Ollama is running and the configured model is available."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_reasoning_optimizer.utils.config import load_config, get_project_root


def main() -> int:
    root = get_project_root()
    cfg = load_config("local_ollama")
    base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
    model = cfg.get("model", "llama3.2")

    try:
        import requests
    except ImportError:
        print("Error: 'requests' is required. pip install requests", file=sys.stderr)
        return 1

    print("Checking Ollama at {} ...".format(base_url), flush=True)
    try:
        r = requests.get("{}/api/tags".format(base_url), timeout=10)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("Ollama is not running or not reachable at {}.".format(base_url), file=sys.stderr)
        print("  Install from https://ollama.com and start Ollama, then run this again.", file=sys.stderr)
        return 1
    except requests.exceptions.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        return 1

    data = r.json()
    models = [m.get("name", "") for m in data.get("models", [])]
    # Ollama tags can be "name:tag" e.g. "llama3.2:latest"
    model_base = model.split(":")[0] if ":" in model else model
    has_model = any(model_base in m or m.startswith(model) for m in models)

    if not has_model:
        print("Model '{}' not found. Available models: {}".format(model, ", ".join(models) or "(none)"), file=sys.stderr)
        print("  Pull it with: ollama pull {}".format(model), file=sys.stderr)
        return 1

    print("Ollama is running and model '{}' is available.".format(model), flush=True)
    print("You can run training with: python scripts/train_reinforce.py --config local_ollama", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
