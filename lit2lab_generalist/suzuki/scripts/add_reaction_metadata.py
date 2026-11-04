#!/usr/bin/env python3
"""Add shared reaction metadata to the Suzuki example data."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUZUKI_DIR = SCRIPT_DIR.parent
REPO_ROOT = SUZUKI_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.reaction_metadata import run_metadata_cli


if __name__ == "__main__":
    run_metadata_cli(SUZUKI_DIR / "data" / "example.parquet")
