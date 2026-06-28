"""Block 1: data processing and train/validation/test artifact generation."""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("data", run_name="__main__")


if __name__ == "__main__":
    main()
