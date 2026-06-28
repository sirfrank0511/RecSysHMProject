"""
End-to-end H&M recommender capstone pipeline.

Stages:
  1. Retrieval: repeat purchase, recent global popularity, age-bucket popularity,
     and item-to-item co-occurrence/category candidates.
  2. Ranking: LightGBM LambdaRank when available, otherwise deterministic scoring.
  3. Reranking: de-duplicate, blend scores, and keep mild category diversity.
  4. Evaluation/submission: MAP@12 offline and Kaggle-format CSV.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


DATA_DIR = "./Data"
OUT_DIR = "./capstone_artifacts"
RANDOM_SEED = 42
TOPK = 12


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: str = DATA_DIR
    out_dir: str = OUT_DIR
    topk: int = TOPK
    popular_k: int = 48
    repeat_k: int = 24
    cooc_k: int = 24
    category_k: int = 24
    train_days: int = 28
    label_days: int = 7
    min_score: float = -1e9
    max_train_users: int = 120_000
    max_eval_users: int = 80_000


def article_to_str(article_id: int) -> str:
    return str(int(article_id)).zfill(10)


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(
        os.path.join(data_dir, "transactions_train.csv"),
        dtype={"customer_id": "string", "article_id": "int32", "sales_channel_id": "int8"},
    )
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["week"] = ((tx["t_dat"] - tx["t_dat"].min()).dt.days // 7).astype("int16")

    articles = pd.read_csv(os.path.join(data_dir, "articles.csv"), dtype={"article_id": "int32"})
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), dtype={"customer_id": "string"})
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"), dtype={"customer_id": "string"})
    return tx, articles, customers, sample


def map_article_features(articles: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "article_id",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]
    out = articles[cols].copy()
    for col in cols[1:]:
        out[col] = out[col].fillna(-1).astype("int32")
    return out


def prepare_customers(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers[["customer_id", "age", "FN", "Active"]].copy()
    out["age"] = out["age"].fillna(out["age"].median()).astype("float32")
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[0, 20, 30, 40, 50, 60, 120],
        labels=False,
        include_lowest=True,
    ).fillna(2).astype("int8")
    out["FN"] = out["FN"].fillna(0).astype("int8")
    out["Active"] = out["Active"].fillna(0).astype("int8")
    return out


def map12(pred: dict[str, list[int]], actual: dict[str, set[int]], k: int = 12) -> float:
    scores = []
    for customer_id, true_items in actual.items():
        if not true_items:
            continue
        hits = 0
        score = 0.0
        seen = set()
        for rank, item in enumerate(pred.get(customer_id, [])[:k], start=1):
            if item in seen:
                continue
            seen.add(item)
            if item in true_items:
                hits += 1
                score += hits / rank
        scores.append(score / min(len(true_items), k))
    return float(np.mean(scores)) if scores else 0.0


def label_dict(label_tx: pd.DataFrame) -> dict[str, set[int]]:
    return label_tx.groupby("customer_id")["article_id"].agg(lambda x: set(map(int, x))).to_dict()


def recent_popularity(tx: pd.DataFrame, cutoff: pd.Timestamp, days: int, k: int) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=days)
    recent = tx[(tx["t_dat"] >= start) & (tx["t_dat"] < cutoff)]
    pop = recent.groupby("article_id").size().rename("pop_count").reset_index()
    pop["pop_rank"] = pop["pop_count"].rank(method="first", ascending=False).astype("int32")
    return pop.sort_values(["pop_count", "article_id"], ascending=[False, True]).head(k).reset_index(drop=True)


def add_candidate_block(
    frames: list[pd.DataFrame],
    users: pd.Series,
    items: Iterable[int],
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


def build_cooc_pairs(history: pd.DataFrame, articles: pd.DataFrame, k: int) -> pd.DataFrame:
    recent_pairs = history.sort_values(["customer_id", "t_dat"]).groupby("customer_id").tail(6)
    pairs = recent_pairs.merge(recent_pairs, on="customer_id", suffixes=("_src", "_cand"))
    pairs = pairs[pairs["article_id_src"] != pairs["article_id_cand"]]
    if pairs.empty:
        return pd.DataFrame(columns=["article_id_src", "article_id", "cooc_rank"])
    counts = pairs.groupby(["article_id_src", "article_id_cand"]).size().rename("cooc_count").reset_index()
    counts = counts.sort_values(["article_id_src", "cooc_count"], ascending=[True, False])
    counts["cooc_rank"] = counts.groupby("article_id_src").cumcount() + 1
    counts = counts[counts["cooc_rank"] <= k]
    return counts.rename(columns={"article_id_cand": "article_id"})[["article_id_src", "article_id", "cooc_rank"]]


def build_candidates(
    tx: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    target_customers: pd.Series,
    cutoff: pd.Timestamp,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    history = tx[tx["t_dat"] < cutoff].copy()
    target_customers = target_customers.drop_duplicates().astype("string")
    frames: list[pd.DataFrame] = []

    pop = recent_popularity(history, cutoff, cfg.train_days, cfg.popular_k)
    add_candidate_block(frames, target_customers, pop["article_id"], "global_pop", 0.20)

    # Repeat-purchase retrieval with stronger recency score.
    user_recent = history[history["customer_id"].isin(target_customers)]
    user_recent = user_recent.sort_values(["customer_id", "t_dat"]).groupby("customer_id").tail(cfg.repeat_k)
    if not user_recent.empty:
        repeat = user_recent[["customer_id", "article_id", "t_dat"]].drop_duplicates(["customer_id", "article_id"], keep="last")
        repeat["days_since"] = (cutoff - repeat["t_dat"]).dt.days.clip(lower=0)
        repeat["source"] = "repeat"
        repeat["source_score"] = 1.0 / (1.0 + repeat["days_since"])
        frames.append(repeat[["customer_id", "article_id", "source", "source_score"]])

    cust = prepare_customers(customers)
    hist_age = history.merge(cust[["customer_id", "age_bin"]], on="customer_id", how="left")
    age_pop = (
        hist_age[hist_age["t_dat"] >= cutoff - pd.Timedelta(days=cfg.train_days)]
        .groupby(["age_bin", "article_id"])
        .size()
        .rename("age_pop_count")
        .reset_index()
        .sort_values(["age_bin", "age_pop_count"], ascending=[True, False])
    )
    age_pop["age_rank"] = age_pop.groupby("age_bin").cumcount() + 1
    age_pop = age_pop[age_pop["age_rank"] <= cfg.popular_k]
    target_age = target_customers.to_frame("customer_id").merge(cust[["customer_id", "age_bin"]], on="customer_id", how="left")
    age_block = target_age.merge(age_pop[["age_bin", "article_id", "age_rank"]], on="age_bin", how="left").dropna(subset=["article_id"])
    if not age_block.empty:
        age_block["article_id"] = age_block["article_id"].astype("int32")
        age_block["source"] = "age_pop"
        age_block["source_score"] = 0.15 + 1.0 / (age_block["age_rank"].astype("float32") + 5.0)
        frames.append(age_block[["customer_id", "article_id", "source", "source_score"]])

    # Co-occurrence from each user's latest bought items.
    cooc = build_cooc_pairs(history[history["t_dat"] >= cutoff - pd.Timedelta(days=cfg.train_days)], articles, cfg.cooc_k)
    if not cooc.empty and not user_recent.empty:
        seeds = user_recent[["customer_id", "article_id"]].rename(columns={"article_id": "article_id_src"})
        cooc_block = seeds.merge(cooc, on="article_id_src", how="inner")
        cooc_block["source"] = "cooc"
        cooc_block["source_score"] = 0.12 + 1.0 / (cooc_block["cooc_rank"].astype("float32") + 10.0)
        frames.append(cooc_block[["customer_id", "article_id", "source", "source_score"]])

    # Category retrieval: popular products in categories the user recently bought.
    art = articles[["article_id", "product_type_no", "garment_group_no"]].copy()
    recent_with_cat = history[history["t_dat"] >= cutoff - pd.Timedelta(days=cfg.train_days)].merge(art, on="article_id", how="left")
    cat_pop = (
        recent_with_cat.groupby(["product_type_no", "garment_group_no", "article_id"])
        .size()
        .rename("cat_pop_count")
        .reset_index()
        .sort_values(["product_type_no", "garment_group_no", "cat_pop_count"], ascending=[True, True, False])
    )
    cat_pop["cat_rank"] = cat_pop.groupby(["product_type_no", "garment_group_no"]).cumcount() + 1
    cat_pop = cat_pop[cat_pop["cat_rank"] <= cfg.category_k]
    user_cat = user_recent.merge(art, on="article_id", how="left")[["customer_id", "product_type_no", "garment_group_no"]].drop_duplicates()
    cat_block = user_cat.merge(cat_pop, on=["product_type_no", "garment_group_no"], how="inner")
    if not cat_block.empty:
        cat_block["source"] = "category_pop"
        cat_block["source_score"] = 0.10 + 1.0 / (cat_block["cat_rank"].astype("float32") + 20.0)
        frames.append(cat_block[["customer_id", "article_id", "source", "source_score"]])

    candidates = pd.concat(frames, ignore_index=True)
    source_dummies = pd.get_dummies(candidates["source"], prefix="src")
    candidates = pd.concat([candidates[["customer_id", "article_id", "source_score"]], source_dummies], axis=1)
    agg = {"source_score": "max", **{c: "max" for c in source_dummies.columns}}
    return candidates.groupby(["customer_id", "article_id"], as_index=False).agg(agg)


def add_features(
    candidates: pd.DataFrame,
    tx: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    history = tx[tx["t_dat"] < cutoff]
    recent = history[history["t_dat"] >= cutoff - pd.Timedelta(days=cfg.train_days)]
    candidate_users = candidates["customer_id"].drop_duplicates()
    candidate_items = candidates["article_id"].drop_duplicates()

    item_recent = recent[recent["article_id"].isin(candidate_items)]
    item_stats = item_recent.groupby("article_id").agg(
        item_pop_28d=("article_id", "size"),
        item_last_seen=("t_dat", "max"),
        item_avg_price=("price", "mean"),
    ).reset_index()
    item_stats["item_days_since"] = (cutoff - item_stats["item_last_seen"]).dt.days.astype("float32")
    item_stats = item_stats.drop(columns=["item_last_seen"])

    user_history = history[history["customer_id"].isin(candidate_users)]
    user_stats = user_history.groupby("customer_id").agg(
        user_tx_count=("article_id", "size"),
        user_unique_items=("article_id", "nunique"),
        user_last_seen=("t_dat", "max"),
        user_avg_price=("price", "mean"),
    ).reset_index()
    user_stats["user_days_since"] = (cutoff - user_stats["user_last_seen"]).dt.days.astype("float32")
    user_stats = user_stats.drop(columns=["user_last_seen"])

    pair_items = candidate_items
    user_item_history = user_history[user_history["article_id"].isin(pair_items)]
    user_item = user_item_history.groupby(["customer_id", "article_id"]).agg(
        user_item_count=("article_id", "size"),
        user_item_last=("t_dat", "max"),
    ).reset_index()
    user_item["user_item_days_since"] = (cutoff - user_item["user_item_last"]).dt.days.astype("float32")
    user_item = user_item.drop(columns=["user_item_last"])

    feat = candidates.merge(item_stats, on="article_id", how="left")
    feat = feat.merge(user_stats, on="customer_id", how="left")
    feat = feat.merge(user_item, on=["customer_id", "article_id"], how="left")
    feat = feat.merge(map_article_features(articles), on="article_id", how="left")
    feat = feat.merge(prepare_customers(customers), on="customer_id", how="left")

    fill_zero = ["item_pop_28d", "user_tx_count", "user_unique_items", "user_item_count"]
    for col in fill_zero:
        feat[col] = feat[col].fillna(0).astype("float32")
    for col in ["item_days_since", "user_days_since", "user_item_days_since"]:
        feat[col] = feat[col].fillna(999).astype("float32")
    for col in ["item_avg_price", "user_avg_price"]:
        feat[col] = feat[col].fillna(feat[col].median()).astype("float32")
    feat["price_gap"] = (feat["item_avg_price"] - feat["user_avg_price"]).abs().astype("float32")
    feat["log_item_pop_28d"] = np.log1p(feat["item_pop_28d"]).astype("float32")
    feat["log_user_tx_count"] = np.log1p(feat["user_tx_count"]).astype("float32")
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"customer_id", "article_id", "label"}
    return [c for c in df.columns if c not in ignored and pd.api.types.is_numeric_dtype(df[c])]


def attach_labels(features: pd.DataFrame, labels: dict[str, set[int]]) -> pd.DataFrame:
    label_pairs = [(u, i) for u, items in labels.items() for i in items]
    lab = pd.DataFrame(label_pairs, columns=["customer_id", "article_id"])
    lab["label"] = 1
    out = features.merge(lab, on=["customer_id", "article_id"], how="left")
    out["label"] = out["label"].fillna(0).astype("int8")
    return out


def train_ranker(train_df: pd.DataFrame, cols: list[str]):
    if lgb is None:
        return None
    train_df = train_df.sort_values("customer_id")
    group = train_df.groupby("customer_id", sort=False).size().to_numpy()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[cols], train_df["label"], group=group)
    return model


def deterministic_score(df: pd.DataFrame) -> np.ndarray:
    return (
        3.0 * df.get("src_repeat", 0)
        + 1.4 * df.get("src_global_pop", 0)
        + 1.2 * df.get("src_age_pop", 0)
        + 1.1 * df.get("src_cooc", 0)
        + 0.9 * df.get("src_category_pop", 0)
        + df["source_score"].astype("float32")
        + 0.08 * df["log_item_pop_28d"].astype("float32")
        - 0.015 * df["user_item_days_since"].clip(0, 120).astype("float32")
    ).to_numpy(dtype=np.float32)


def rerank(df: pd.DataFrame, score_col: str, topk: int) -> dict[str, list[int]]:
    predictions: dict[str, list[int]] = {}
    sort_cols = ["customer_id", score_col, "source_score", "item_pop_28d"]
    ranked = df.sort_values(sort_cols, ascending=[True, False, False, False])
    for customer_id, grp in ranked.groupby("customer_id", sort=False):
        chosen: list[int] = []
        garment_counts: dict[int, int] = {}
        backup: list[int] = []
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


def sample_users(users: pd.Series, max_users: int) -> pd.Series:
    users = users.drop_duplicates().astype("string")
    if max_users and len(users) > max_users:
        return users.sample(max_users, random_state=RANDOM_SEED)
    return users


def run_offline(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    tx, articles, customers, _ = load_data(cfg.data_dir)
    max_date = tx["t_dat"].max()
    eval_cutoff = max_date - pd.Timedelta(days=cfg.label_days)
    train_cutoff = eval_cutoff - pd.Timedelta(days=cfg.label_days)

    train_labels_tx = tx[(tx["t_dat"] >= train_cutoff) & (tx["t_dat"] < eval_cutoff)]
    eval_labels_tx = tx[(tx["t_dat"] >= eval_cutoff) & (tx["t_dat"] <= max_date)]

    train_users = sample_users(train_labels_tx["customer_id"], cfg.max_train_users)
    eval_users = sample_users(eval_labels_tx["customer_id"], cfg.max_eval_users)

    print(f"Train cutoff={train_cutoff.date()} label window < {eval_cutoff.date()} users={len(train_users)}")
    train_cand = build_candidates(tx, articles, customers, train_users, train_cutoff, cfg)
    train_feat = add_features(train_cand, tx, articles, customers, train_cutoff, cfg)
    train_df = attach_labels(train_feat, label_dict(train_labels_tx))
    print("Training candidates:", train_df.shape, "positive_rate:", round(float(train_df["label"].mean()), 5))

    cols = feature_columns(train_df)
    model = train_ranker(train_df, cols)

    print(f"Eval cutoff={eval_cutoff.date()} label window <= {max_date.date()} users={len(eval_users)}")
    eval_cand = build_candidates(tx, articles, customers, eval_users, eval_cutoff, cfg)
    eval_feat = add_features(eval_cand, tx, articles, customers, eval_cutoff, cfg)
    if model is not None:
        eval_feat["rank_score"] = model.predict(eval_feat[cols])
        model.booster_.save_model(os.path.join(cfg.out_dir, "lgbm_ranker.txt"))
    else:
        eval_feat["rank_score"] = deterministic_score(eval_feat)

    preds = rerank(eval_feat, "rank_score", cfg.topk)
    score = map12(preds, label_dict(eval_labels_tx), cfg.topk)
    print(f"MAP@{cfg.topk}: {score:.5f}")

    pd.DataFrame({"metric": [f"MAP@{cfg.topk}"], "value": [score]}).to_csv(
        os.path.join(cfg.out_dir, "offline_metrics.csv"), index=False
    )


def run_submission(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    tx, articles, customers, sample = load_data(cfg.data_dir)
    cutoff = tx["t_dat"].max() + pd.Timedelta(days=1)
    target_users = sample["customer_id"].astype("string")
    print(f"Submission cutoff={cutoff.date()} users={len(target_users)}")

    candidates = build_candidates(tx, articles, customers, target_users, cutoff, cfg)
    features = add_features(candidates, tx, articles, customers, cutoff, cfg)

    model_path = os.path.join(cfg.out_dir, "lgbm_ranker.txt")
    if lgb is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        cols = booster.feature_name()
        features["rank_score"] = booster.predict(features[cols])
    else:
        features["rank_score"] = deterministic_score(features)

    preds = rerank(features, "rank_score", cfg.topk)
    global_fill = recent_popularity(tx, cutoff, cfg.train_days, cfg.popular_k)["article_id"].astype(int).tolist()

    rows = []
    for customer_id in target_users:
        recs = preds.get(str(customer_id), [])
        for item in global_fill:
            if item not in recs:
                recs.append(item)
            if len(recs) >= cfg.topk:
                break
        rows.append((customer_id, " ".join(article_to_str(x) for x in recs[: cfg.topk])))

    out = pd.DataFrame(rows, columns=["customer_id", "prediction"])
    path = os.path.join(cfg.out_dir, "submission.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H&M recommender capstone pipeline")
    parser.add_argument("--mode", choices=["offline", "submission"], default="offline")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--max_train_users", type=int, default=120_000)
    parser.add_argument("--max_eval_users", type=int, default=80_000)
    parser.add_argument("--popular_k", type=int, default=48)
    parser.add_argument("--repeat_k", type=int, default=24)
    parser.add_argument("--cooc_k", type=int, default=24)
    parser.add_argument("--category_k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_train_users=args.max_train_users,
        max_eval_users=args.max_eval_users,
        popular_k=args.popular_k,
        repeat_k=args.repeat_k,
        cooc_k=args.cooc_k,
        category_k=args.category_k,
    )
    if args.mode == "offline":
        run_offline(cfg)
    else:
        run_submission(cfg)


if __name__ == "__main__":
    main()
