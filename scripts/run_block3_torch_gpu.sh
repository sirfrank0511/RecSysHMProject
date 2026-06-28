#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-metal/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in this shell; run from Terminal where mps_available=True.")
PY

python blocks/block3_retrieval_train.py \
  --device mps \
  --steps 5000 \
  --batch_size 512 \
  --log_every 100

python blocks/block4_retrieval_eval.py \
  --device mps \
  --sample_users 20000 \
  --k 50
