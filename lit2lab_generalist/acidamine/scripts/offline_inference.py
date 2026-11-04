#!/usr/bin/env python3
"""Run shared SGLang offline inference for acid–amine data."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ACIDAMINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = ACIDAMINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.offline_inference import run_inference_cli


if __name__ == "__main__":
    run_inference_cli(
        ACIDAMINE_DIR / "data" / "example_grpo.parquet",
        ACIDAMINE_DIR / "outputs" / "inference" / "example_predictions.parquet",
    )
