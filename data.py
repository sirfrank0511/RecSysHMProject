# BLOCK 1: DATA PROCESSING
# OBJECTIVES: LOAD DATA, TIME SPLIT, ID MAPPING, USER HISTORIES RETRIEVAL
# SETUP:
#       Training: Up to 2020-08-23
#       Validation: 2020-08-24 to 2020-09-06
#       Deployment Testing: 2020-09-07 to 2020-09-21



import os
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd

# ====== CONFIG ======
DATA_DIR = "./Data"
TRAIN_END = "2020-08-23"
VAL_END = "2020-09-06"
MAX_HIST_LEN = 50
PAD_ITEM_IDX = 0  # reserve 0 for padding; item_idx start from 1
POP_K = 12        # top-K popularity fallback for cold-start
# ====================

def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["article_id"] = tx["article_id"].astype(np.int64)
    tx["customer_id"] = tx["customer_id"].astype(str)

    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    articles = pd.read_csv(os.path.join(DATA_DIR, "articles.csv"))
    return tx, customers, articles

def time_split_transactions(tx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    train = tx[tx["t_dat"] <= train_end].copy()
    val = tx[(tx["t_dat"] > train_end) & (tx["t_dat"] <= val_end)].copy()
    test = tx[tx["t_dat"] > val_end].copy()
    return train, val, test

def build_id_maps(train_tx: pd.DataFrame):
    user_ids = pd.Index(train_tx["customer_id"]).unique()
    item_ids = pd.Index(train_tx["article_id"]).unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {int(a): (i + 1) for i, a in enumerate(item_ids)}  # 1-based; 0 is PAD

    idx2user = user_ids.to_numpy()
    idx2item_article_id = item_ids.to_numpy(dtype=np.int64)
    return user2idx, item2idx, idx2user, idx2item_article_id

def map_ids(tx: pd.DataFrame, user2idx: Dict[str, int], item2idx: Dict[int, int]) -> pd.DataFrame:
    out = tx.copy()
    out["user_idx"] = out["customer_id"].map(user2idx)
    out["item_idx"] = out["article_id"].map(item2idx)
    out = out.dropna(subset=["user_idx", "item_idx"])
    out["user_idx"] = out["user_idx"].astype(np.int32)
    out["item_idx"] = out["item_idx"].astype(np.int32)
    return out

def build_histories_and_next_item_pairs(train_m: pd.DataFrame, num_users: int):
    """
    Theoretically correct next-item training:
      For each user and each event t:
        history = items strictly before t (last MAX_HIST_LEN, right-aligned padded)
        positive = item at t
    Also returns FINAL user state up to TRAIN_END.
    """
    train_m = train_m.sort_values(["user_idx", "t_dat"])

    user_hist_seq_final = np.full((num_users, MAX_HIST_LEN), PAD_ITEM_IDX, dtype=np.int32)
    user_hist_len_final = np.zeros((num_users,), dtype=np.int32)

    u_pos: list[int] = []
    i_pos: list[int] = []
    hist_seq_list: list[np.ndarray] = []
    hist_len_list: list[int] = []

    for u, grp in train_m.groupby("user_idx", sort=False):
        items = grp["item_idx"].to_numpy(dtype=np.int32)
        if len(items) < 2:
            continue

        # Only need recent window; still "correct" because history is capped at MAX_HIST_LEN.
        start = max(1, len(items) - (MAX_HIST_LEN + 1))
        for t in range(start, len(items)):
            hist = items[max(0, t - MAX_HIST_LEN):t]
            if len(hist) == 0:
                continue

            h = np.full((MAX_HIST_LEN,), PAD_ITEM_IDX, dtype=np.int32)
            l = int(len(hist))
            h[-l:] = hist

            u_pos.append(int(u))
            i_pos.append(int(items[t]))
            hist_seq_list.append(h)
            hist_len_list.append(l)

        # Final state at TRAIN_END (for serving / validation)
        hist_final = items[-MAX_HIST_LEN:]
        l_final = int(len(hist_final))
        user_hist_seq_final[int(u), -l_final:] = hist_final
        user_hist_len_final[int(u)] = l_final

    retrieval_u_pos = np.asarray(u_pos, dtype=np.int32)
    retrieval_i_pos = np.asarray(i_pos, dtype=np.int32)
    retrieval_hist_seq = np.stack(hist_seq_list).astype(np.int32) if hist_seq_list else np.zeros((0, MAX_HIST_LEN), np.int32)
    retrieval_hist_len = np.asarray(hist_len_list, dtype=np.int32) if hist_len_list else np.zeros((0,), np.int32)

    return (
        user_hist_seq_final,
        user_hist_len_final,
        retrieval_u_pos,
        retrieval_i_pos,
        retrieval_hist_seq,
        retrieval_hist_len,
    )

def build_window_labels(window_m: pd.DataFrame) -> Dict[int, Set[int]]:
    labels: Dict[int, Set[int]] = {}
    for u, grp in window_m.groupby("user_idx"):
        labels[int(u)] = set(map(int, grp["item_idx"].tolist()))
    return labels

def dict_to_pairs(user2items: dict):
    u_list, i_list = [], []
    for u, items in user2items.items():
        for it in items:
            u_list.append(u)
            i_list.append(it)
    return np.asarray(u_list, dtype=np.int32), np.asarray(i_list, dtype=np.int32)

def run_block1():
    tx, customers, articles = load_raw()
    train_tx, val_tx, test_tx = time_split_transactions(tx)

    user2idx, item2idx, idx2user, idx2item_article_id = build_id_maps(train_tx)
    train_m = map_ids(train_tx, user2idx, item2idx)
    val_m = map_ids(val_tx, user2idx, item2idx)
    test_m = map_ids(test_tx, user2idx, item2idx)

    num_users = len(idx2user)
    num_items_including_pad = max(item2idx.values()) + 1

    (
        user_hist_seq_final,
        user_hist_len_final,
        u_pos,
        i_pos,
        retrieval_hist_seq,
        retrieval_hist_len,
    ) = build_histories_and_next_item_pairs(train_m, num_users)

    val_labels = build_window_labels(val_m)
    test_labels = build_window_labels(test_m)

    # popularity fallback from TRAIN ONLY
    pop_counts = train_m["item_idx"].value_counts()
    pop_items = pop_counts.index.to_numpy(dtype=np.int32)[:POP_K]

    artifacts = {
        "train_m": train_m,
        "val_m": val_m,
        "test_m": test_m,
        "customers": customers,
        "articles": articles,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item_article_id": idx2item_article_id,
        "num_users": num_users,
        "num_items_including_pad": num_items_including_pad,
        "user_hist_seq_final": user_hist_seq_final,
        "user_hist_len_final": user_hist_len_final,
        "retrieval_u_pos": u_pos,
        "retrieval_i_pos": i_pos,
        "retrieval_hist_seq": retrieval_hist_seq,
        "retrieval_hist_len": retrieval_hist_len,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "pop_items": pop_items,
    }
    return artifacts

if __name__ == "__main__":
    artifacts = run_block1()

    print("Block1 done.")
    print("Training Cut-off:", TRAIN_END, "Validation Cut-off:", VAL_END)
    print("train_m rows:", len(artifacts["train_m"]))
    print("val_m rows:", len(artifacts["val_m"]))
    print("test_m rows:", len(artifacts["test_m"]))
    print("num_users:", artifacts["num_users"])
    print("num_items_including_pad:", artifacts["num_items_including_pad"])
    print("user_hist_seq_final shape:", artifacts["user_hist_seq_final"].shape)
    print("retrieval_hist_seq shape:", artifacts["retrieval_hist_seq"].shape)

    print("val unique users (mapped):", artifacts["val_m"]["user_idx"].nunique() if len(artifacts["val_m"]) else 0)
    print("test unique users (mapped):", artifacts["test_m"]["user_idx"].nunique() if len(artifacts["test_m"]) else 0)

    # DIAG: per-example pos-in-history (should be low-ish; not near 1.0)
    rng = np.random.default_rng(0)
    n = len(artifacts["retrieval_u_pos"])
    if n:
        idx = rng.integers(0, n, size=min(200_000, n))
        ii = artifacts["retrieval_i_pos"][idx]
        hh = artifacts["retrieval_hist_seq"][idx]
        pos_in_hist = np.mean(np.any(hh == ii[:, None], axis=1))
        print("[BLOCK1 CHECK] train pos-in-history (per-example hist):", float(pos_in_hist))
        print("[BLOCK1 CHECK] num train pairs:", n)

    os.makedirs("./artifacts_block1", exist_ok=True)

    np.save("./artifacts_block1/idx2item_article_id.npy", artifacts["idx2item_article_id"])
    np.save("./artifacts_block1/num_items_including_pad.npy", np.array([artifacts["num_items_including_pad"]], dtype=np.int32))
    np.save("./artifacts_block1/num_users.npy", np.array([artifacts["num_users"]], dtype=np.int32))

    # final user state (for serving)
    np.save("./artifacts_block1/user_hist_seq_final.npy", artifacts["user_hist_seq_final"])
    np.save("./artifacts_block1/user_hist_len_final.npy", artifacts["user_hist_len_final"])

    # retrieval training tuples
    np.save("./artifacts_block1/retrieval_u_pos.npy", artifacts["retrieval_u_pos"])
    np.save("./artifacts_block1/retrieval_i_pos.npy", artifacts["retrieval_i_pos"])
    np.save("./artifacts_block1/retrieval_hist_seq.npy", artifacts["retrieval_hist_seq"])
    np.save("./artifacts_block1/retrieval_hist_len.npy", artifacts["retrieval_hist_len"])
    print("[BLOCK1] Saved retrieval_hist_seq:", artifacts["retrieval_hist_seq"].shape)

    # train stream for other blocks if needed
    train_item_idx_full = artifacts["train_m"]["item_idx"].to_numpy(dtype=np.int32)
    np.save("./artifacts_block1/train_item_idx_full.npy", train_item_idx_full)
    print("[BLOCK1] Saved train_item_idx_full:", train_item_idx_full.shape)

    # popularity fallback
    np.save("./artifacts_block1/pop_items.npy", artifacts["pop_items"])
    print("[BLOCK1] Saved pop_items:", artifacts["pop_items"].shape, artifacts["pop_items"])

    # val/test pairs
    val_u_idx, val_i_idx = dict_to_pairs(artifacts["val_labels"])
    np.save("./artifacts_block1/val_u_idx.npy", val_u_idx)
    np.save("./artifacts_block1/val_i_idx.npy", val_i_idx)
    print("Saved val pairs:", val_u_idx.shape, val_i_idx.shape)

    test_u_idx, test_i_idx = dict_to_pairs(artifacts["test_labels"])
    np.save("./artifacts_block1/test_u_idx.npy", test_u_idx)
    np.save("./artifacts_block1/test_i_idx.npy", test_i_idx)
    print("Saved test pairs:", test_u_idx.shape, test_i_idx.shape)

    print("Saved Block 1 artifacts to ./artifacts_block1/")
