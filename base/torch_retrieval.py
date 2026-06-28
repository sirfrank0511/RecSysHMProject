"""
PyTorch two-tower retrieval for the H&M capstone.

It uses Apple Silicon MPS when PyTorch can initialize it, CUDA if available,
otherwise CPU.
Training examples come from artifacts_block1 produced by data.py.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK1_DIR = "./artifacts_block1"
BLOCK2_DIR = "./artifacts_block2"
OUT_DIR = "./artifacts_torch"
IMG_EMB_PATH = os.path.join(BLOCK2_DIR, "item_image_emb_mobilenetv3small.npy")
PAD_ITEM = 0


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


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class TorchTwoTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        image_emb: np.ndarray | None,
        emb_dim: int = 128,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_images = image_emb is not None
        self.item_id_emb = nn.Embedding(num_items, emb_dim, padding_idx=PAD_ITEM)

        if self.use_images:
            img = torch.from_numpy(image_emb.astype(np.float32))
            self.register_buffer("image_emb", img)
            img_dim = int(img.shape[1])
            self.user_img_proj = nn.Sequential(
                nn.Linear(img_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.item_img_proj = nn.Sequential(
                nn.Linear(img_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.user_fuse = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.item_fuse = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.user_fuse = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            self.item_fuse = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def encode_users(self, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist != PAD_ITEM).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        id_vec = (self.item_id_emb(hist) * mask.unsqueeze(-1)).sum(dim=1) / denom

        if self.use_images:
            img = self.image_emb[hist]
            img_vec = (img * mask.unsqueeze(-1)).sum(dim=1) / denom
            img_vec = self.user_img_proj(l2norm(img_vec))
            out = self.user_fuse(torch.cat([id_vec, img_vec], dim=-1))
        else:
            out = self.user_fuse(id_vec)
        return l2norm(out)

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_vec = self.item_id_emb(item_idx)
        if self.use_images:
            img_vec = self.item_img_proj(self.image_emb[item_idx])
            out = self.item_fuse(torch.cat([id_vec, img_vec], dim=-1))
        else:
            out = self.item_fuse(id_vec)
        return l2norm(out)

    def loss(self, hist: torch.Tensor, pos_item: torch.Tensor) -> tuple[torch.Tensor, float]:
        u = self.encode_users(hist)
        v = self.encode_items(pos_item)
        logits = (u @ v.T) / self.temperature

        # Duplicate positive items in a batch create false negatives. Mask duplicate columns.
        eye = torch.eye(pos_item.shape[0], dtype=torch.bool, device=pos_item.device)
        dup = (pos_item[:, None] == pos_item[None, :]) & ~eye
        logits = logits.masked_fill(dup, -1e9)
        labels = torch.arange(pos_item.shape[0], device=pos_item.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            rank = (logits > logits.diag().unsqueeze(1)).sum(dim=1) + 1
            recall10 = (rank <= 10).float().mean().item()
        return loss, recall10


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 512
    steps: int = 5000
    lr: float = 1e-3
    emb_dim: int = 128
    temperature: float = 0.2
    log_every: int = 100
    seed: int = 42
    use_images: bool = True
    device: str = "auto"


def load_model(cfg: TrainConfig) -> tuple[TorchTwoTower, torch.device]:
    num_items = int(np.load(os.path.join(BLOCK1_DIR, "num_items_including_pad.npy"))[0])
    image_emb = None
    if cfg.use_images and os.path.exists(IMG_EMB_PATH):
        image_emb = np.load(IMG_EMB_PATH, mmap_mode="r")
    device = choose_device(cfg.device)
    model = TorchTwoTower(num_items=num_items, image_emb=image_emb, emb_dim=cfg.emb_dim, temperature=cfg.temperature)
    model.to(device)
    return model, device


def train(cfg: TrainConfig) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    hist_seq = np.load(os.path.join(BLOCK1_DIR, "retrieval_hist_seq.npy"), mmap_mode="r")
    pos_items = np.load(os.path.join(BLOCK1_DIR, "retrieval_i_pos.npy"), mmap_mode="r")
    n = int(pos_items.shape[0])

    model, device = load_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-6)

    print("device:", device)
    print("train examples:", n)
    print("params:", sum(p.numel() for p in model.parameters()))

    model.train()
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, n, size=cfg.batch_size)
        hist = torch.as_tensor(np.asarray(hist_seq[idx]), dtype=torch.long, device=device)
        pos = torch.as_tensor(np.asarray(pos_items[idx]), dtype=torch.long, device=device)

        opt.zero_grad(set_to_none=True)
        loss, recall10 = model.loss(hist, pos)
        loss.backward()
        opt.step()

        if step == 1 or step % cfg.log_every == 0:
            print(f"step={step} loss={loss.item():.4f} inbatch_recall@10={recall10:.4f}")

    path = os.path.join(OUT_DIR, "two_tower_torch.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, path)
    print("saved:", path)


def compute_item_embeddings(model: TorchTwoTower, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    num_items = int(np.load(os.path.join(BLOCK1_DIR, "num_items_including_pad.npy"))[0])

    model.eval()
    item_vecs = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            ids = torch.arange(start, min(start + batch_size, num_items), dtype=torch.long, device=device)
            item_vecs.append(model.encode_items(ids).cpu().numpy())
    item_vecs = np.vstack(item_vecs).astype(np.float32)
    item_vecs[PAD_ITEM] = 0.0
    np.save(os.path.join(OUT_DIR, "item_vecs_torch.npy"), item_vecs)
    return item_vecs


def compute_user_embeddings(
    model: TorchTwoTower,
    device: torch.device,
    users: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    user_hist = np.load(os.path.join(BLOCK1_DIR, "user_hist_seq_final.npy"), mmap_mode="r")

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            hist = torch.as_tensor(np.asarray(user_hist[users[start:end]]), dtype=torch.long, device=device)
            out.append(model.encode_users(hist).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def topk_search(user_vecs: np.ndarray, item_vecs: np.ndarray, k: int, user_batch: int = 1024, item_chunk: int = 20000) -> np.ndarray:
    out = np.zeros((len(user_vecs), k), dtype=np.int32)
    item_t = torch.as_tensor(item_vecs)
    for u0 in range(0, len(user_vecs), user_batch):
        u1 = min(u0 + user_batch, len(user_vecs))
        u = torch.as_tensor(np.asarray(user_vecs[u0:u1]), dtype=torch.float32)
        best_scores = torch.full((u1 - u0, k), -1e9)
        best_items = torch.zeros((u1 - u0, k), dtype=torch.long)
        for i0 in range(0, item_vecs.shape[0], item_chunk):
            chunk = item_t[i0 : i0 + item_chunk]
            scores = u @ chunk.T
            sc, ix = torch.topk(scores, k=k, dim=1)
            ix = ix + i0
            merged_scores = torch.cat([best_scores, sc], dim=1)
            merged_items = torch.cat([best_items, ix], dim=1)
            best_scores, pos = torch.topk(merged_scores, k=k, dim=1)
            best_items = torch.gather(merged_items, 1, pos)
        out[u0:u1] = best_items.numpy().astype(np.int32)
    return out


def map_at_k(users: np.ndarray, recs: np.ndarray, val_u: np.ndarray, val_i: np.ndarray, k: int = 12) -> float:
    labels: dict[int, set[int]] = {}
    for u, i in zip(val_u, val_i):
        labels.setdefault(int(u), set()).add(int(i))
    scores = []
    for u, row in zip(users, recs):
        true = labels.get(int(u), set())
        if not true:
            continue
        hits = 0
        score = 0.0
        seen = set()
        for rank, item in enumerate(row[:k], start=1):
            item = int(item)
            if item in seen:
                continue
            seen.add(item)
            if item in true:
                hits += 1
                score += hits / rank
        scores.append(score / min(len(true), k))
    return float(np.mean(scores)) if scores else 0.0


def evaluate(device_name: str = "auto", sample_users: int = 20000, k: int = 50) -> None:
    ckpt = torch.load(os.path.join(OUT_DIR, "two_tower_torch.pt"), map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    model, device = load_model(TrainConfig(**{**cfg.__dict__, "device": device_name}))
    model.load_state_dict(ckpt["state_dict"])
    print("device:", device)

    val_u = np.load(os.path.join(BLOCK1_DIR, "val_u_idx.npy"))
    val_i = np.load(os.path.join(BLOCK1_DIR, "val_i_idx.npy"))
    users = np.array(sorted(set(map(int, val_u.tolist()))), dtype=np.int32)
    if sample_users and len(users) > sample_users:
        rng = np.random.default_rng(42)
        users = rng.choice(users, size=sample_users, replace=False)
    item_vecs = compute_item_embeddings(model, device)
    user_vecs = compute_user_embeddings(model, device, users)
    recs = topk_search(user_vecs, item_vecs, k=k)
    np.save(os.path.join(OUT_DIR, "retrieved_users_val_torch.npy"), users)
    np.save(os.path.join(OUT_DIR, "retrieved_topk_val_torch.npy"), recs)
    score = map_at_k(users, recs, val_u, val_i, k=12)
    np.savez(os.path.join(OUT_DIR, "retrieval_eval_metrics_torch.npz"), map12=score, n_eval_users=len(users))
    print(f"MAP@12: {score:.5f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch two-tower retrieval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--batch_size", type=int, default=512)
    tr.add_argument("--steps", type=int, default=5000)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--emb_dim", type=int, default=128)
    tr.add_argument("--temperature", type=float, default=0.2)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--no_images", action="store_true")

    ev = sub.add_parser("eval")
    ev.add_argument("--device", default="auto")
    ev.add_argument("--sample_users", type=int, default=20000)
    ev.add_argument("--k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
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
    else:
        evaluate(device_name=args.device, sample_users=args.sample_users, k=args.k)


if __name__ == "__main__":
    main()
