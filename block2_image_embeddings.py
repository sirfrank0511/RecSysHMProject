"""Block 2: image embedding generation aligned to Block 1 item ids."""

from __future__ import annotations

from image_embeddings import precompute_image_embeddings
import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 2: precompute image embeddings.")
    parser.add_argument("--run", action="store_true", help="Actually compute embeddings.")
    parser.add_argument("--images_dir", default="./Data/images")
    parser.add_argument("--out_dir", default="./artifacts_block2")
    parser.add_argument("--backbone", default="mobilenetv3small", choices=["mobilenetv3small", "efficientnetb0"])
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    idx_path = "./artifacts_block1/idx2item_article_id.npy"
    num_path = "./artifacts_block1/num_items_including_pad.npy"
    out_path = os.path.join(args.out_dir, f"item_image_emb_{args.backbone}.npy")
    mask_path = os.path.join(args.out_dir, f"item_has_image_{args.backbone}.npy")

    if not args.run:
        for path in [out_path, mask_path]:
            print(path, "exists=", os.path.exists(path))
        if os.path.exists(out_path) and os.path.exists(mask_path):
            emb = np.load(out_path, mmap_mode="r")
            mask = np.load(mask_path)
            print("embedding shape:", emb.shape, emb.dtype)
            print("image coverage:", float(mask.mean()))
        print("Use --run to recompute embeddings.")
        return

    idx2item_article_id = np.load(idx_path)
    num_items = int(np.load(num_path)[0])
    precompute_image_embeddings(
        idx2item_article_id=idx2item_article_id,
        num_items_including_pad=num_items,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
