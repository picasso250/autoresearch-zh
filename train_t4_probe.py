"""
Thin wrapper around train_t4.py for quick T4 capacity probing.

Example:
    python train_t4_probe.py --dataset tinystorieszh --depth 8 --model-dim 640 --device-batch-size 8 --max-steps 50
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    train_script = Path(__file__).with_name("train_t4.py")
    cmd = [sys.executable, str(train_script), *sys.argv[1:], "--skip-eval"]
    print(">", " ".join(cmd), flush=True)
    env = os.environ.copy()
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
