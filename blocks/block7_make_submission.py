"""Block 7: create Kaggle submission.csv from the trained hybrid ranker.

This deployment block uses the full transaction history as context, generates
business/domain candidates in batches, scores them with the Block 5 ranker, and
writes Kaggle's required customer_id,prediction CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.capstone_recommender import (
    PipelineConfig,
    add_features,
    article_to_str,
    build_cooc_pairs,
    load_data,
    prepare_customers,
    recent_popularity,
)


RANKER_DIR = "./artifacts_ranker"
SUBMISSION_DIR = "./submissions"


def add_candidate_block(
    frames: list[pd.DataFrame],
    users: pd.Series,
    items: pd.Series,
    source: str,
    base_score: float,
) -> None:
    item_list = list(map(int, items))
    if not item_list:
        return
    block = pd.MultiIndex.from_product([users.astype("string"), item_list], names=["customer_id", "article_id"]).to_frame(index=False)
    block["source"] = source
    block["source_score"] = base_score
    frames.append(block)


def load_feature_names(path: str) -> list[str]:
    return pd.read_csv(path, header=None)[0].tolist()


def align_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0
    return out[cols]


def fast_rerank_batch(df: pd.DataFrame, score_col: str, topk: int) -> dict[str, list[int]]:
    predictions: dict[str, list[int]] = {}
    for customer_id, grp in df.groupby("customer_id", sort=False):
        grp = grp.sort_values([score_col, "source_score", "item_pop_28d"], ascending=[False, False, False])
        chosen = []
        garment_counts: dict[int, int] = {}
        backup = []
        for row in grp.itertuples(index=False):
            item = int(row.article_id)
            garment = int(getattr(row, "garment_group_no", -1))
            if len(chosen) < topk and garment_counts.get(garment, 0) < 4:
                chosen.append(item)
                garment_counts[garment] = garment_counts.get(garment, 0) + 1
            else:
                backup.append(item)
            if len(chosen) >= topk:
                break
        if len(chosen) < topk:
            for item in backup:
                if item not in chosen:
                    chosen.append(item)
                if len(chosen) >= topk:
                    break
        predictions[str(customer_id)] = chosen[:topk]
    return predictions


def precompute_candidate_context(
    tx: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    history = tx[tx["t_dat"] < cutoff].copy()
    recent = history[history["t_dat"] >= cutoff - pd.Timedelta(days=cfg.train_days)]
    cust = prepare_customers(customers)
    art = articles[["article_id", "product_type_no", "garment_group_no"]].copy()

    pop = recent_popularity(history, cutoff, cfg.train_days, cfg.popular_k)

    hist_age = recent.merge(cust[["customer_id", "age_bin"]], on="customer_id", how="left")
    age_pop = (
        hist_age.groupby(["age_bin", "article_id"])
        .size()
        .rename("age_pop_count")
        .reset_index()
        .sort_values(["age_bin", "age_pop_count"], ascending=[True, False])
    )
    age_pop["age_rank"] = age_pop.groupby("age_bin").cumcount() + 1
    age_pop = age_pop[age_pop["age_rank"] <= cfg.popular_k]

    cooc = build_cooc_pairs(recent, articles, cfg.cooc_k)

    recent_with_cat = recent.merge(art, on="article_id", how="left")
    cat_pop = (
        recent_with_cat.groupby(["product_type_no", "garment_group_no", "article_id"])
        .size()
        .rename("cat_pop_count")
        .reset_index()
        .sort_values(["product_type_no", "garment_group_no", "cat_pop_count"], ascending=[True, True, False])
    )
    cat_pop["cat_rank"] = cat_pop.groupby(["product_type_no", "garment_group_no"]).cumcount() + 1
    cat_pop = cat_pop[cat_pop["cat_rank"] <= cfg.category_k]

    return {
        "history": history,
        "pop": pop,
        "cust": cust,
        "art": art,
        "age_pop": age_pop,
        "cooc": cooc,
        "cat_pop": cat_pop,
    }


def build_submission_candidates(batch_users: pd.Series, cutoff: pd.Timestamp, cfg: PipelineConfig, ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    history = ctx["history"]
    target_customers = batch_users.drop_duplicates().astype("string")
    frames: list[pd.DataFrame] = []

    add_candidate_block(frames, target_customers, ctx["pop"]["article_id"], "global_pop", 0.20)

    user_recent = history[history["customer_id"].isin(target_customers)]
    user_recent = user_recent.sort_values(["customer_id", "t_dat"]).groupby("customer_id").tail(cfg.repeat_k)
    if not user_recent.empty:
        repeat = user_recent[["customer_id", "article_id", "t_dat"]].drop_duplicates(["customer_id", "article_id"], keep="last")
        repeat["days_since"] = (cutoff - repeat["t_dat"]).dt.days.clip(lower=0)
        repeat["source"] = "repeat"
        repeat["source_score"] = 1.0 / (1.0 + repeat["days_since"])
        frames.append(repeat[["customer_id", "article_id", "source", "source_score"]])

    target_age = target_customers.to_frame("customer_id").merge(ctx["cust"][["customer_id", "age_bin"]], on="customer_id", how="left")
    age_block = target_age.merge(ctx["age_pop"][["age_bin", "article_id", "age_rank"]], on="age_bin", how="left").dropna(subset=["article_id"])
    if not age_block.empty:
        age_block["article_id"] = age_block["article_id"].astype("int32")
        age_block["source"] = "age_pop"
        age_block["source_score"] = 0.15 + 1.0 / (age_block["age_rank"].astype("float32") + 5.0)
        frames.append(age_block[["customer_id", "article_id", "source", "source_score"]])

    if not ctx["cooc"].empty and not user_recent.empty:
        seeds = user_recent[["customer_id", "article_id"]].rename(columns={"article_id": "article_id_src"})
        cooc_block = seeds.merge(ctx["cooc"], on="article_id_src", how="inner")
        cooc_block["source"] = "cooc"
        cooc_block["source_score"] = 0.12 + 1.0 / (cooc_block["cooc_rank"].astype("float32") + 10.0)
        frames.append(cooc_block[["customer_id", "article_id", "source", "source_score"]])

    if not user_recent.empty:
        user_cat = user_recent.merge(ctx["art"], on="article_id", how="left")[["customer_id", "product_type_no", "garment_group_no"]].drop_duplicates()
        cat_block = user_cat.merge(ctx["cat_pop"], on=["product_type_no", "garment_group_no"], how="inner")
        if not cat_block.empty:
            cat_block["source"] = "category_pop"
            cat_block["source_score"] = 0.10 + 1.0 / (cat_block["cat_rank"].astype("float32") + 20.0)
            frames.append(cat_block[["customer_id", "article_id", "source", "source_score"]])

    candidates = pd.concat(frames, ignore_index=True)
    source_dummies = pd.get_dummies(candidates["source"], prefix="src")
    candidates = pd.concat([candidates[["customer_id", "article_id", "source_score"]], source_dummies], axis=1)
    agg = {"source_score": "max", **{c: "max" for c in source_dummies.columns}}
    return candidates.groupby(["customer_id", "article_id"], as_index=False).agg(agg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 7: make Kaggle submission.")
    parser.add_argument("--data_dir", default="./Data")
    parser.add_argument("--ranker_dir", default=RANKER_DIR)
    parser.add_argument("--out_dir", default=SUBMISSION_DIR)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--max_customers", type=int, default=0, help="Optional debug limit for the sample submission rows.")
    parser.add_argument("--popular_k", type=int, default=48)
    parser.add_argument("--repeat_k", type=int, default=24)
    parser.add_argument("--cooc_k", type=int, default=24)
    parser.add_argument("--category_k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = os.path.join(args.ranker_dir, "lgbm_ranker.txt")
    feature_path = os.path.join(args.ranker_dir, "ranker_features.txt")
    if not os.path.exists(model_path):
        raise SystemExit(f"Missing ranker model: {model_path}. Run Block 5 first.")
    if not os.path.exists(feature_path):
        raise SystemExit(f"Missing ranker feature list: {feature_path}. Run Block 5 first.")

    tx, articles, customers, sample = load_data(args.data_dir)
    cutoff = tx["t_dat"].max() + pd.Timedelta(days=1)
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        popular_k=args.popular_k,
        repeat_k=args.repeat_k,
        cooc_k=args.cooc_k,
        category_k=args.category_k,
    )

    booster = lgb.Booster(model_file=model_path)
    feature_cols = load_feature_names(feature_path)
    global_fill = recent_popularity(tx, cutoff, cfg.train_days, max(args.popular_k, 24))["article_id"].astype(int).tolist()
    ctx = precompute_candidate_context(tx, articles, customers, cutoff, cfg)

    target_users = sample["customer_id"].astype("string")
    if args.max_customers:
        target_users = target_users.iloc[: args.max_customers]
    path = os.path.join(args.out_dir, "submission.csv")
    wrote_header = False
    for start in range(0, len(target_users), args.batch_size):
        end = min(start + args.batch_size, len(target_users))
        batch_users = target_users.iloc[start:end]
        print(f"submission batch: {start}..{end - 1}", flush=True)

        candidates = build_submission_candidates(batch_users, cutoff, cfg, ctx)
        candidates["retrieval_rank"] = 9999
        candidates["src_torch_retrieval"] = 0
        features = add_features(candidates, tx, articles, customers, cutoff, cfg)
        features["rank_score"] = booster.predict(align_features(features, feature_cols))
        preds = fast_rerank_batch(features, "rank_score", cfg.topk)

        rows: list[tuple[str, str]] = []
        for customer_id in batch_users:
            recs = preds.get(str(customer_id), [])
            for item in global_fill:
                if item not in recs:
                    recs.append(item)
                if len(recs) >= cfg.topk:
                    break
            rows.append((str(customer_id), " ".join(article_to_str(x) for x in recs[: cfg.topk])))

        out = pd.DataFrame(rows, columns=["customer_id", "prediction"])
        out.to_csv(path, index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True

    print("saved:", path)
    print("rows:", len(target_users))


if __name__ == "__main__":
    main()
