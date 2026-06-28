"""Block 6: rerank ranked candidates and compute final validation/list-quality metrics."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from capstone_recommender import map12, rerank
from data import TRAIN_END, build_id_maps, map_ids, time_split_transactions
from capstone_recommender import load_data


RANKER_DIR = "./artifacts_ranker"


def item_metadata(articles: pd.DataFrame) -> pd.DataFrame:
    cols = ["article_id", "product_type_no", "section_no", "garment_group_no"]
    out = articles[cols].copy()
    for col in cols[1:]:
        out[col] = out[col].fillna(-1).astype("int32")
    return out


def recommendation_rows(preds: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for customer_id, items in preds.items():
        for rank, article_id in enumerate(items, start=1):
            rows.append((customer_id, int(article_id), rank))
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "rank"])


def relevance_metrics(preds: dict[str, list[int]], labels: dict[str, set[int]], topk: int) -> dict[str, float]:
    hits = []
    recalls = []
    reciprocal_ranks = []
    for customer_id, true_items in labels.items():
        if not true_items:
            continue
        recs = preds.get(str(customer_id), [])[:topk]
        seen = set()
        hit_count = 0
        first_hit_rank = None
        for rank, item in enumerate(recs, start=1):
            if item in seen:
                continue
            seen.add(item)
            if item in true_items:
                hit_count += 1
                if first_hit_rank is None:
                    first_hit_rank = rank
        hits.append(1.0 if hit_count > 0 else 0.0)
        recalls.append(hit_count / len(true_items))
        reciprocal_ranks.append(0.0 if first_hit_rank is None else 1.0 / first_hit_rank)
    return {
        f"HitRate@{topk}": float(np.mean(hits)) if hits else 0.0,
        f"Recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
        f"MRR@{topk}": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def personalization_metric(preds: dict[str, list[int]], topk: int, n_pairs: int = 20000) -> float:
    users = list(preds.keys())
    if len(users) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    sims = []
    for _ in range(min(n_pairs, len(users) * 20)):
        u1, u2 = rng.choice(users, size=2, replace=False)
        s1 = set(preds[u1][:topk])
        s2 = set(preds[u2][:topk])
        union = len(s1 | s2)
        sims.append(0.0 if union == 0 else len(s1 & s2) / union)
    return float(1.0 - np.mean(sims)) if sims else 0.0


def repeat_item_share(preds: dict[str, list[int]], train_tx: pd.DataFrame, topk: int) -> float:
    history = train_tx.groupby("customer_id")["article_id"].agg(lambda x: set(map(int, x))).to_dict()
    shares = []
    for customer_id, recs in preds.items():
        bought = history.get(str(customer_id), set())
        if not recs:
            continue
        shares.append(sum(1 for item in recs[:topk] if int(item) in bought) / min(len(recs), topk))
    return float(np.mean(shares)) if shares else 0.0


def segmented_relevance_metrics(
    preds: dict[str, list[int]],
    labels: dict[str, set[int]],
    train_tx: pd.DataFrame,
    topk: int,
) -> dict[str, float]:
    counts = train_tx.groupby("customer_id").size()
    segments = {
        "cold_1_3": counts[(counts >= 1) & (counts <= 3)].index.astype(str),
        "light_4_10": counts[(counts >= 4) & (counts <= 10)].index.astype(str),
        "medium_11_50": counts[(counts >= 11) & (counts <= 50)].index.astype(str),
        "heavy_51_plus": counts[counts >= 51].index.astype(str),
    }
    out = {}
    for name, users in segments.items():
        user_set = set(users)
        seg_labels = {u: items for u, items in labels.items() if str(u) in user_set}
        if not seg_labels:
            continue
        seg = relevance_metrics(preds, seg_labels, topk)
        out[f"MAP@{topk}_{name}"] = map12(preds, seg_labels, topk)
        out[f"HitRate@{topk}_{name}"] = seg[f"HitRate@{topk}"]
        out[f"Recall@{topk}_{name}"] = seg[f"Recall@{topk}"]
    return out


def list_quality_metrics(
    preds: dict[str, list[int]],
    tx: pd.DataFrame,
    articles: pd.DataFrame,
    cutoff: pd.Timestamp,
    topk: int,
) -> dict[str, float]:
    recs = recommendation_rows(preds)
    if recs.empty:
        return {
            "catalog_coverage": 0.0,
            "avg_unique_product_types": 0.0,
            "avg_unique_garment_groups": 0.0,
            "avg_unique_sections": 0.0,
            "avg_intra_list_product_type_diversity": 0.0,
            "avg_intra_list_garment_group_diversity": 0.0,
            "new_item_share_30d": 0.0,
            "new_item_share_90d": 0.0,
            "avg_item_popularity_percentile": 0.0,
        }

    meta = item_metadata(articles)
    recs = recs.merge(meta, on="article_id", how="left")
    history = tx[tx["t_dat"] < cutoff]
    catalog_size = int(history["article_id"].nunique())
    item_first_seen = history.groupby("article_id")["t_dat"].min().rename("first_seen").reset_index()
    item_pop = history.groupby("article_id").size().rename("item_pop_count").reset_index()
    item_pop["item_popularity_percentile"] = item_pop["item_pop_count"].rank(pct=True)
    recs = recs.merge(item_first_seen, on="article_id", how="left")
    recs = recs.merge(item_pop[["article_id", "item_popularity_percentile"]], on="article_id", how="left")
    recs["days_since_first_seen"] = (cutoff - recs["first_seen"]).dt.days

    per_user = recs.groupby("customer_id").agg(
        unique_product_types=("product_type_no", "nunique"),
        unique_garment_groups=("garment_group_no", "nunique"),
        unique_sections=("section_no", "nunique"),
    )

    denom = max(topk, 1)
    return {
        "catalog_coverage": float(recs["article_id"].nunique() / max(catalog_size, 1)),
        "avg_unique_product_types": float(per_user["unique_product_types"].mean()),
        "avg_unique_garment_groups": float(per_user["unique_garment_groups"].mean()),
        "avg_unique_sections": float(per_user["unique_sections"].mean()),
        "avg_intra_list_product_type_diversity": float((per_user["unique_product_types"] / denom).mean()),
        "avg_intra_list_garment_group_diversity": float((per_user["unique_garment_groups"] / denom).mean()),
        "new_item_share_30d": float((recs["days_since_first_seen"] <= 30).mean()),
        "new_item_share_90d": float((recs["days_since_first_seen"] <= 90).mean()),
        "avg_item_popularity_percentile": float(recs["item_popularity_percentile"].fillna(0).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 6: rerank and evaluate.")
    parser.add_argument("--data_dir", default="./Data")
    parser.add_argument("--ranker_dir", default=RANKER_DIR)
    parser.add_argument("--topk", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked_path = os.path.join(args.ranker_dir, "val_ranked_candidates.pkl")
    if not os.path.exists(ranked_path):
        raise SystemExit(f"Missing ranked candidates: {ranked_path}. Run Block 5 first.")

    tx, articles, _, _ = load_data(args.data_dir)
    train_tx, val_tx, _ = time_split_transactions(tx)
    user2idx, item2idx, _, _ = build_id_maps(train_tx)
    val_m = map_ids(val_tx, user2idx, item2idx)
    labels = val_m.groupby("customer_id")["article_id"].agg(lambda x: set(map(int, x))).to_dict()

    ranked = pd.read_pickle(ranked_path)
    eval_users = set(ranked["customer_id"].astype(str).unique().tolist())
    labels = {str(u): items for u, items in labels.items() if str(u) in eval_users}
    preds = rerank(ranked, "rank_score", args.topk)
    score = map12(preds, labels, args.topk)
    cutoff = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    relevance = relevance_metrics(preds, labels, args.topk)
    quality = list_quality_metrics(preds, tx, articles, cutoff, args.topk)
    behavior = {
        "personalization": personalization_metric(preds, args.topk),
        "repeat_item_share": repeat_item_share(preds, train_tx, args.topk),
    }
    segments = segmented_relevance_metrics(preds, labels, train_tx, args.topk)
    metrics = {f"MAP@{args.topk}": score, **relevance, **quality, **behavior, **segments}
    out = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    metrics_path = os.path.join(args.ranker_dir, "rerank_eval_metrics.csv")
    out.to_csv(metrics_path, index=False)
    print(f"MAP@{args.topk}: {score:.5f}")
    for name, value in {**relevance, **quality, **behavior, **segments}.items():
        print(f"{name}: {value:.5f}")
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
