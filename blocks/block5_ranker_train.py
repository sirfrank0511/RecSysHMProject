"""Block 5: train ranker on hybrid retrieval candidates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.capstone_recommender import (
    PipelineConfig,
    add_features,
    attach_labels,
    build_candidates,
    feature_columns,
    label_dict,
    load_data,
    train_ranker,
)
from base.data import TRAIN_END, build_id_maps, map_ids, time_split_transactions


OUT_DIR = "./artifacts_ranker"
TORCH_DIR = "./artifacts_torch"


def build_mapping_frames(tx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train_tx, val_tx, _ = time_split_transactions(tx)
    user2idx, item2idx, idx2user, idx2item_article_id = build_id_maps(train_tx)
    val_m = map_ids(val_tx, user2idx, item2idx)
    return idx2user, idx2item_article_id, train_tx, val_m


def load_torch_candidates(max_users: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    users = np.load(os.path.join(TORCH_DIR, "retrieved_users_val_torch.npy"))
    topk = np.load(os.path.join(TORCH_DIR, "retrieved_topk_val_torch.npy"))
    if max_users and len(users) > max_users:
        users = users[:max_users]
        topk = topk[:max_users]

    rows = []
    for row, user_idx in enumerate(users):
        for rank, item_idx in enumerate(topk[row], start=1):
            rows.append((int(user_idx), int(item_idx), rank))
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "retrieval_rank"]), users


def merge_candidate_sources(torch_candidates: pd.DataFrame, heuristic_candidates: pd.DataFrame) -> pd.DataFrame:
    all_cols = sorted((set(torch_candidates.columns) | set(heuristic_candidates.columns)) - {"customer_id", "article_id"})
    frames = []
    for frame in [torch_candidates, heuristic_candidates]:
        frame = frame.copy()
        for col in all_cols:
            if col not in frame.columns:
                frame[col] = 0
        frames.append(frame[["customer_id", "article_id", *all_cols]])

    merged = pd.concat(frames, ignore_index=True)
    agg = {}
    for col in all_cols:
        agg[col] = "min" if col == "retrieval_rank" else "max"
    out = merged.groupby(["customer_id", "article_id"], as_index=False).agg(agg)
    if "retrieval_rank" in out.columns:
        out["retrieval_rank"] = out["retrieval_rank"].replace(0, 9999).astype("float32")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 5: train ranker from retrieval candidates.")
    parser.add_argument("--data_dir", default="./Data")
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--max_users", type=int, default=0, help="Optional small-sample debug limit.")
    parser.add_argument("--popular_k", type=int, default=48)
    parser.add_argument("--repeat_k", type=int, default=24)
    parser.add_argument("--cooc_k", type=int, default=24)
    parser.add_argument("--category_k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tx, articles, customers, _ = load_data(args.data_dir)
    idx2user, idx2item_article_id, _, val_m = build_mapping_frames(tx)
    cutoff = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        popular_k=args.popular_k,
        repeat_k=args.repeat_k,
        cooc_k=args.cooc_k,
        category_k=args.category_k,
    )

    torch_candidates, user_idx = load_torch_candidates(max_users=args.max_users or None)
    torch_candidates = torch_candidates[torch_candidates["item_idx"] > 0].copy()
    torch_candidates["customer_id"] = idx2user[torch_candidates["user_idx"].to_numpy()]
    torch_candidates["article_id"] = idx2item_article_id[torch_candidates["item_idx"].to_numpy() - 1]
    torch_candidates["source_score"] = 1.0 / torch_candidates["retrieval_rank"].astype("float32")
    torch_candidates["src_torch_retrieval"] = 1
    torch_candidates = torch_candidates[
        ["customer_id", "article_id", "source_score", "src_torch_retrieval", "retrieval_rank"]
    ]

    target_customers = pd.Series(idx2user[user_idx], dtype="string")
    heuristic_candidates = build_candidates(tx, articles, customers, target_customers, cutoff, cfg)
    heuristic_candidates["retrieval_rank"] = 9999

    candidates = merge_candidate_sources(torch_candidates, heuristic_candidates)
    print("torch candidates:", torch_candidates.shape)
    print("heuristic candidates:", heuristic_candidates.shape)
    print("hybrid candidates:", candidates.shape)

    feat = add_features(candidates, tx, articles, customers, cutoff, cfg)

    labels_original = val_m.copy()
    eval_customers = set(target_customers.astype(str).tolist())
    labels_original = labels_original[labels_original["customer_id"].isin(eval_customers)]
    train_df = attach_labels(feat, label_dict(labels_original[["customer_id", "article_id"]]))
    cols = feature_columns(train_df)

    print("ranker rows:", train_df.shape)
    print("positive rate:", round(float(train_df["label"].mean()), 6))
    model = train_ranker(train_df, cols)
    if model is None:
        raise SystemExit("LightGBM is unavailable; cannot train Block 5 ranker.")

    model_path = os.path.join(args.out_dir, "lgbm_ranker.txt")
    model.booster_.save_model(model_path)
    pd.Series(cols).to_csv(os.path.join(args.out_dir, "ranker_features.txt"), index=False, header=False)
    train_df["rank_score"] = model.predict(train_df[cols])
    train_df.to_pickle(os.path.join(args.out_dir, "val_ranked_candidates.pkl"))
    print("saved:", model_path)
    print("saved:", os.path.join(args.out_dir, "val_ranked_candidates.pkl"))


if __name__ == "__main__":
    main()
