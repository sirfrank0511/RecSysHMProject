"""Block 3: train PyTorch two-tower retrieval model."""

from __future__ import annotations

import argparse

from torch_retrieval import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 3: train retrieval.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            emb_dim=args.emb_dim,
            temperature=args.temperature,
            log_every=args.log_every,
            use_images=not args.no_images,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
