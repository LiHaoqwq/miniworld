"""One-shot: collect data, train three world models, evaluate, write metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script), *(extra or [])]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    steps = sys.argv[1] if len(sys.argv) > 1 else "800"
    run("train.py", ["--model", "all", "--steps", steps])
    run("eval.py")
    print("\nMiniWorld finished. See results/metrics.json and results/rollout_strip.png")
