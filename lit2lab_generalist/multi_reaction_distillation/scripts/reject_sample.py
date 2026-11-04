#!/usr/bin/env python3
"""Reject-sample an explicitly supplied inference parquet."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.reject_sampling import run_reject_sampling_cli


if __name__ == "__main__":
    run_reject_sampling_cli()
