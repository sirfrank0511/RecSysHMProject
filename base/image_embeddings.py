"""Block 2 implementation: create image embeddings with PyTorch/torchvision."""

from __future__ import annotations

import argparse
import glob
import importlib
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def article_id_to_image_path(images_dir: str, article_id: int) -> str:
    """Map H&M article id to images/0xx/0xxxxxxxxx.jpg."""
    s = str(int(article_id)).zfill(10)
    return os.path.join(images_dir, s[:3], f"{s}.jpg")


def choose_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


@dataclass(frozen=True)
class ImageModelSpec:
    model: nn.Module
    dim: int
    image_size: int


def build_image_model(backbone: str = "mobilenetv3small", pretrained: bool = True) -> ImageModelSpec:
    """Return a feature extractor that outputs pooled image vectors."""
    backbone = backbone.lower()
    if backbone == "mobilenetv3small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        dim = int(model.classifier[0].in_features)
        model.classifier = nn.Identity()
        image_size = 224
    elif backbone == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        dim = int(model.classifier[1].in_features)
        model.classifier = nn.Identity()
        image_size = 224
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    model.eval()
    return ImageModelSpec(model=model, dim=dim, image_size=image_size)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: np.ndarray, image_size: int):
        self.image_paths = image_paths.astype(str)
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return int(len(self.image_paths))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        if not path:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32), torch.tensor(0, dtype=torch.int8)
        try:
            with Image.open(path) as img:
                x = self.transform(img.convert("RGB"))
            return x, torch.tensor(1, dtype=torch.int8)
        except Exception:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32), torch.tensor(0, dtype=torch.int8)


def build_image_paths(idx2item_article_id: np.ndarray, num_items_including_pad: int, images_dir: str) -> np.ndarray:
    image_paths = np.full((num_items_including_pad,), "", dtype=object)
    for j, article_id in enumerate(idx2item_article_id):
        item_idx = j + 1
        path = article_id_to_image_path(images_dir, int(article_id))
        if os.path.exists(path):
            image_paths[item_idx] = path
    return image_paths


def precompute_image_embeddings(
    idx2item_article_id: np.ndarray,
    num_items_including_pad: int,
    images_dir: str,
    out_dir: str,
    backbone: str = "mobilenetv3small",
    batch_size: int = 256,
    device: str = "auto",
    pretrained: bool = True,
    max_items: int | None = None,
):
    """Precompute image embeddings aligned to item_idx.

    The output shape is `(num_items_including_pad, embedding_dim)`, with row 0
    reserved as the zero PAD vector. Missing/unreadable images also receive zero
    vectors and `has_image=0`.
    """
    os.makedirs(out_dir, exist_ok=True)
    image_paths = build_image_paths(idx2item_article_id, num_items_including_pad, images_dir)
    if max_items:
        image_paths = image_paths[:max_items]
        num_items_including_pad = len(image_paths)

    spec = build_image_model(backbone=backbone, pretrained=pretrained)
    run_device = choose_device(device)
    model = spec.model.to(run_device)

    dataset = ImagePathDataset(image_paths, image_size=spec.image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    img_emb = np.zeros((num_items_including_pad, spec.dim), dtype=np.float32)
    has_image = np.zeros((num_items_including_pad,), dtype=np.int8)

    print("device:", run_device)
    print("backbone:", backbone, "pretrained:", pretrained)
    print("embedding dim:", spec.dim)
    print("items:", num_items_including_pad)

    offset = 0
    with torch.no_grad():
        for batch_imgs, batch_valid in loader:
            batch_imgs = batch_imgs.to(run_device)
            feats = model(batch_imgs).detach().cpu().numpy().astype(np.float32)
            b = feats.shape[0]
            valid = batch_valid.numpy().astype(np.int8)
            feats[valid == 0] = 0.0
            img_emb[offset : offset + b] = feats
            has_image[offset : offset + b] = valid
            offset += b
            if offset % (batch_size * 50) == 0 or offset >= num_items_including_pad:
                print(f"processed {offset}/{num_items_including_pad}")

    img_emb[0, :] = 0.0
    has_image[0] = 0

    emb_path = os.path.join(out_dir, f"item_image_emb_{backbone}.npy")
    mask_path = os.path.join(out_dir, f"item_has_image_{backbone}.npy")
    np.save(emb_path, img_emb)
    np.save(mask_path, has_image)

    print("saved:", emb_path)
    print("saved:", mask_path)
    print("embedding shape:", img_emb.shape, "has_image rate:", float(has_image.mean()))
    return emb_path, mask_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 2: precompute PyTorch image embeddings aligned to item_idx.")
    parser.add_argument("--run", action="store_true", help="Actually run embedding precompute.")
    parser.add_argument("--use_block1", action="store_true", help="Run Block 1 instead of loading saved artifacts.")
    parser.add_argument("--block1_module", default="base.data", help="Module that defines run_block1().")
    parser.add_argument("--idx2item_npy", default="./artifacts_block1/idx2item_article_id.npy")
    parser.add_argument("--num_items_npy", default="./artifacts_block1/num_items_including_pad.npy")
    parser.add_argument("--images_dir", default="./Data/images")
    parser.add_argument("--out_dir", default="./artifacts_block2")
    parser.add_argument("--backbone", default="mobilenetv3small", choices=["mobilenetv3small", "efficientnetb0"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet weights.")
    parser.add_argument("--max_items", type=int, default=0, help="Debug limit; 0 means all items.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run:
        print(
            "\n[Block2] Not running because --run was not provided.\n\n"
            "Typical usage:\n"
            "  python blocks/block2_image_embeddings.py --run --device mps\n"
        )
        raise SystemExit(0)

    if not os.path.exists(args.images_dir):
        raise SystemExit(f"ERROR: images_dir does not exist: {args.images_dir}")

    subdirs = [d for d in glob.glob(os.path.join(args.images_dir, "*")) if os.path.isdir(d)]
    if len(subdirs) < 10:
        print(f"WARNING: {args.images_dir} has only {len(subdirs)} subfolders. Double-check images_dir.")

    if args.use_block1:
        mod = importlib.import_module(args.block1_module)
        artifacts = mod.run_block1()
        idx2item_article_id = artifacts["idx2item_article_id"]
        num_items_including_pad = int(artifacts["num_items_including_pad"])
    else:
        idx2item_article_id = np.load(args.idx2item_npy)
        num_items_including_pad = int(np.load(args.num_items_npy)[0])

    precompute_image_embeddings(
        idx2item_article_id=idx2item_article_id,
        num_items_including_pad=num_items_including_pad,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        max_items=args.max_items or None,
    )


if __name__ == "__main__":
    main()
