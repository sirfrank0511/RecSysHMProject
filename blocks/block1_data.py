"""Block 1: data processing and train/validation/test artifact generation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    runpy.run_module("base.data", run_name="__main__")


if __name__ == "__main__":
    main()
