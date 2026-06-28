# BLOCK 2: IMAGE EMBEDDINGS
# OBJECTIVES: CREATE IMAGE EMBEDDINGS FOR ARTICLES



import os
from typing import Tuple
import numpy as np
import tensorflow as tf
import argparse
import glob
import importlib

# Connect Image Path to Specific Article
def article_id_to_image_path(images_dir: str, article_id: int) -> str:
    # NOTE: H&M images are stored like:
    # images/0xx/0xxxxxxxxx.jpg
    # Example: article_id=123456789 -> images/012/0123456789.jpg

    s = str(int(article_id)).zfill(10)
    sub = s[:3]
    return os.path.join(images_dir, sub, f"{s}.jpg")

# Build Image Model
def build_image_model(backbone: str = "mobilenetv3small") -> Tuple[tf.keras.Model, int, tuple]:
    # Returns model (a pooled feature vector), dim (embedding dimension), image_size (image dimension (h and w))
    backbone = backbone.lower()
    if backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg"
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        image_size = (224, 224)
        dim = base.output_shape[-1]  # 1280
    elif backbone == "mobilenetv3small":
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet", pooling="avg"
        )
        preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
        image_size = (224, 224)
        dim = base.output_shape[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    inp = tf.keras.Input(shape=(image_size[0], image_size[1], 3), dtype=tf.float32)
    x = preprocess(inp)
    out = base(x)
    model = tf.keras.Model(inp, out)
    return model, dim, image_size

# Create Embedding Dataset
def make_dataset(image_paths: np.ndarray, image_size: tuple, batch_size: int):
    """
    image_paths: array of strings length N (N = num_items_including_pad)
      - may contain "" for PAD or missing images
    """

    def _load(path):
        # path: scalar tf.string
        # if path == "" => return zeros image and valid=0
        is_empty = tf.equal(path, "")
        def load_real():
            bytes_ = tf.io.read_file(path)
            img = tf.image.decode_jpeg(bytes_, channels=3)
            img = tf.image.resize(img, image_size, antialias=True)
            img = tf.cast(img, tf.float32)
            return img, tf.constant(1, tf.int32)

        def load_zero():
            img = tf.zeros((image_size[0], image_size[1], 3), tf.float32)
            return img, tf.constant(0, tf.int32)

        img, valid = tf.cond(is_empty, load_zero, load_real)
        return img, valid

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Aggregate Run of Block2
def precompute_image_embeddings(
    idx2item_article_id: np.ndarray,
    num_items_including_pad: int,
    images_dir: str,
    out_dir: str,
    backbone: str = "efficientnetb0",
    batch_size: int = 256,
):
    """
    idx2item_article_id: shape (I,) list of original article_id values for items seen in train
                         and corresponds to item_idx 1..I in the SAME ORDER used by item2idx.
                         (From Block 1 build_id_maps)

    We build image_paths array of length num_items_including_pad:
      - image_paths[0] = "" for PAD
      - image_paths[item_idx] = filepath for that item_idx
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---- Build path array aligned to item_idx ----
    image_paths = np.full((num_items_including_pad,), "", dtype=object)
    image_paths[0] = ""  # PAD

    # item_idx = position+1
    for j, article_id in enumerate(idx2item_article_id):
        item_idx = j + 1
        p = article_id_to_image_path(images_dir, int(article_id))
        if os.path.exists(p):
            image_paths[item_idx] = p
        else:
            image_paths[item_idx] = ""  # missing -> zero vector

    # ---- Build model ----
    model, dim, image_size = build_image_model(backbone=backbone)
    model.trainable = False

    # Optional speed: mixed precision on GPU
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ds = make_dataset(image_paths.astype(str), image_size, batch_size)

    # ---- Allocate outputs ----
    img_emb = np.zeros((num_items_including_pad, dim), dtype=np.float32)
    has_image = np.zeros((num_items_including_pad,), dtype=np.int8)

    # ---- Run batches ----
    idx = 0
    for batch_imgs, batch_valid in ds:
        feats = model(batch_imgs, training=False)
        feats = tf.cast(feats, tf.float32).numpy()  # ensure float32 for saving

        b = feats.shape[0]
        img_emb[idx:idx+b] = feats
        has_image[idx:idx+b] = batch_valid.numpy().astype(np.int8)
        idx += b

        if idx % (batch_size * 50) == 0:
            print(f"Processed {idx}/{num_items_including_pad} items...")

    # ---- Ensure PAD is zero ----
    img_emb[0, :] = 0.0
    has_image[0] = 0

    # ---- Save ----
    emb_path = os.path.join(out_dir, f"item_image_emb_{backbone}.npy")
    mask_path = os.path.join(out_dir, f"item_has_image_{backbone}.npy")

    np.save(emb_path, img_emb)
    np.save(mask_path, has_image)

    print("Saved:", emb_path)
    print("Saved:", mask_path)
    print("Embedding shape:", img_emb.shape, "has_image rate:", has_image.mean())

    return emb_path, mask_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 2: Precompute image embeddings aligned to item_idx."
    )

    parser.add_argument("--run", action="store_true", help="Actually run embedding precompute.")

    # Mode Selection
    parser.add_argument(
        "--use_block1",
        action="store_true",
        help="Run Block 1 (import module + call run_block1) instead of loading saved artifacts."
    )
    parser.add_argument(
        "--block1_module",
        type=str,
        default="block1_data",
        help="Module that defines run_block1() (default: block1_data)."
    )

    # Artifacts (default path)
    parser.add_argument(
        "--idx2item_npy",
        type=str,
        default="./artifacts_block1/idx2item_article_id.npy",
        help="Path to idx2item_article_id.npy saved from Block 1."
    )
    parser.add_argument(
        "--num_items_npy",
        type=str,
        default="./artifacts_block1/num_items_including_pad.npy",
        help="Path to num_items_including_pad.npy saved from Block 1."
    )

    # Dataset Layout Defaults
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./Data/images",
        help="Path to images/ directory (default: ./Data/images)."
    )
    parser.add_argument("--out_dir", type=str, default="./artifacts_block2")

    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3small",
        choices=["mobilenetv3small", "efficientnetb0"],
    )
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    if not args.run:
        print(
            "\n[Block2] Not running because --run was not provided.\n\n"
            "Typical (pipeline) usage:\n"
            "  python blocks/block2_image_embeddings.py --run\n\n"
            "One-command convenience (re-runs Block 1):\n"
            "  python blocks/block2_image_embeddings.py --run --use_block1\n"
        )
        raise SystemExit(0)

    if not os.path.exists(args.images_dir):
        raise SystemExit(f"ERROR: images_dir does not exist: {args.images_dir}")

    # Quick sanity check: do we have 0xx subfolders?
    subdirs = [d for d in glob.glob(os.path.join(args.images_dir, "*")) if os.path.isdir(d)]
    if len(subdirs) < 10:
        print(
            f"WARNING: {args.images_dir} has only {len(subdirs)} subfolders. "
            "H&M images usually look like images/0xx/0xxxxxxxxx.jpg. Double-check images_dir."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Acquire Block 1 outputs ----
    if args.use_block1:
        mod = importlib.import_module(args.block1_module)
        if not hasattr(mod, "run_block1"):
            raise SystemExit(f"ERROR: {args.block1_module} has no run_block1()")
        artifacts = mod.run_block1()
        idx2item_article_id = artifacts["idx2item_article_id"]
        num_items_including_pad = int(artifacts["num_items_including_pad"])
        print("[Block2] Using Block 1 via import:", args.block1_module)
    else:
        # Default: load saved artifacts
        for p in [args.idx2item_npy, args.num_items_npy]:
            if not os.path.exists(p):
                raise SystemExit(
                    f"ERROR: Missing Block 1 artifact: {p}\n"
                    "Run Block 1 first (and save artifacts), or run Block 2 with --use_block1."
                )
        idx2item_article_id = np.load(args.idx2item_npy)
        num_items_including_pad = int(np.load(args.num_items_npy)[0])
        print("[Block2] Loaded Block 1 artifacts from disk.")

    print("[Block2] num_items_including_pad:", num_items_including_pad)
    print("[Block2] items without PAD:", len(idx2item_article_id))
    print("[Block2] images_dir:", args.images_dir)
    print("[Block2] out_dir:", args.out_dir)
    print("[Block2] backbone:", args.backbone, "batch_size:", args.batch_size)

    precompute_image_embeddings(
        idx2item_article_id=idx2item_article_id,
        num_items_including_pad=num_items_including_pad,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )
