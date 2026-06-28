"""Block 4: evaluate PyTorch retrieval candidates with MAP@12."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.torch_retrieval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 4: retrieval evaluation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample_users", type=int, default=20000)
    parser.add_argument("--k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(device_name=args.device, sample_users=args.sample_users, k=args.k)


if __name__ == "__main__":
    main()
