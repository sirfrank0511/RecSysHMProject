# RecSysHMProject

End-to-end recommender-system capstone for the H&M Personalized Fashion Recommendations Kaggle task.

The project follows a production-style recommender architecture:

```text
data processing
-> DNN retrieval
-> hybrid candidate generation
-> supervised ranking
-> reranking
-> validation metrics
-> Kaggle submission
```

## Repository Layout

```text
base/                  Shared implementation modules
blocks/                Block-by-block runnable pipeline scripts
scripts/               Convenience runners
docs/                  Project reasoning and result summaries
archive_tensorflow_legacy/
                       Old legacy files archived for deletion later
requirements-metal.txt Python dependencies for local Apple Silicon/PyTorch setup
```

Generated data, model artifacts, virtual environments, and submissions are intentionally ignored by git. See [Data Management](docs/Data%20Management.md).

## Active Pipeline

| Block | Script | Purpose |
|---|---|---|
| 1 | `blocks/block1_data.py` | Time split, id mapping, histories, labels |
| 2 | `blocks/block2_image_embeddings.py` | Check/recompute item image embeddings |
| 3 | `blocks/block3_retrieval_train.py` | Train PyTorch two-tower DNN retrieval |
| 4 | `blocks/block4_retrieval_eval.py` | Evaluate retrieval and save candidates |
| 5 | `blocks/block5_ranker_train.py` | Train hybrid LightGBM ranker |
| 6 | `blocks/block6_rerank_submit.py` | Rerank and compute validation/list-quality metrics |
| 7 | `blocks/block7_make_submission.py` | Generate Kaggle `submission.csv` |

Shared modules:

| Module | Purpose |
|---|---|
| `base/data.py` | Block 1 implementation |
| `base/image_embeddings.py` | Image embedding logic |
| `base/torch_retrieval.py` | PyTorch two-tower model, training, retrieval eval |
| `base/capstone_recommender.py` | Candidate generation, ranking features, reranking utilities |

## Results

Final validation metrics:

```text
MAP@12: 0.05017
HitRate@12: 0.21395
Recall@12: 0.09467
MRR@12: 0.10907
Personalization: 0.91677
New item share 30d: 0.32128
Catalog coverage: 0.08759
```

Full metric summary:

```text
docs/Results Summary.md
```

Project reasoning:

```text
docs/Underlying Logic.md
```

## Setup

Create/activate the local environment:

```bash
python3.11 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-metal.txt
```

For LightGBM on macOS, `libomp` may be required:

```bash
brew install libomp
```

## Data

The Kaggle dataset is large and is not committed to git.

Expected local layout:

```text
Data/
  articles.csv
  customers.csv
  transactions_train.csv
  sample_submission.csv
  images/
```

See [docs/Data Management.md](docs/Data%20Management.md) for options to recreate or share data/artifacts.

## Running The Pipeline

Activate the environment first:

```bash
source .venv-metal/bin/activate
```

Run block by block:

```bash
python blocks/block1_data.py
python blocks/block2_image_embeddings.py
python blocks/block3_retrieval_train.py --device mps
python blocks/block4_retrieval_eval.py --device mps
python blocks/block5_ranker_train.py
python blocks/block6_rerank_submit.py
python blocks/block7_make_submission.py
```

Convenience runner for PyTorch retrieval on Apple Silicon:

```bash
./scripts/run_block3_torch_gpu.sh
```

Thin pipeline runner:

```bash
python scripts/run_pipeline.py --from_block block5 --to_block block7
```

## Submission

Final submission is generated locally at:

```text
submissions/submission.csv
```

The generated submission was format-checked locally:

- `1,371,980` rows
- columns: `customer_id`, `prediction`
- exactly 12 article ids per row
- customer ids match `sample_submission.csv` order

## Design Summary

The DNN two-tower model is used as the learned retrieval backbone. Its candidates are combined with business/domain candidate generators:

- repeat purchases
- recent popularity
- age-bin popularity
- category popularity
- co-occurrence candidates

The hybrid candidate pool is ranked by LightGBM LambdaRank and reranked into final top-12 lists. This keeps the project DNN-based while reflecting how recommender systems are built in production.
