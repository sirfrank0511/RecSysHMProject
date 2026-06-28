"""Thin block-by-block pipeline runner."""

from __future__ import annotations

import argparse
import subprocess
import sys


BLOCKS = {
    "block1": [sys.executable, "blocks/block1_data.py"],
    "block2": [sys.executable, "blocks/block2_image_embeddings.py", "--run"],
    "block3": [sys.executable, "blocks/block3_retrieval_train.py"],
    "block4": [sys.executable, "blocks/block4_retrieval_eval.py"],
    "block5": [sys.executable, "blocks/block5_ranker_train.py"],
    "block6": [sys.executable, "blocks/block6_rerank_submit.py"],
    "block7": [sys.executable, "blocks/block7_make_submission.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run capstone blocks in order.")
    parser.add_argument("--from_block", choices=BLOCKS, default="block1")
    parser.add_argument("--to_block", choices=BLOCKS, default="block6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = list(BLOCKS)
    start = keys.index(args.from_block)
    end = keys.index(args.to_block)
    if start > end:
        raise SystemExit("--from_block must be before or equal to --to_block")
    for key in keys[start : end + 1]:
        print(f"\n=== {key} ===")
        subprocess.run(BLOCKS[key], check=True)


if __name__ == "__main__":
    main()
