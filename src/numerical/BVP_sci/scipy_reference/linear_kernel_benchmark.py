"""Compatibility entry point for the shared somelinalg SciPy LU runner."""

from __future__ import annotations

import sys
from pathlib import Path

REFERENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "somelinalg"
    / "sparse_lu_reference"
)
sys.path.insert(0, str(REFERENCE_DIR))

from splu_benchmark import main


if __name__ == "__main__":
    main()
