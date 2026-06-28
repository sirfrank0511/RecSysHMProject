# Data Management

The H&M Kaggle dataset and generated artifacts are too large for normal git.

Local sizes observed during this project:

```text
Data/                 ~61 GB
artifacts_block1/     ~4.6 GB
artifacts_block2/     ~221 MB
artifacts_torch/      ~325 MB
artifacts_ranker/     ~1.8 GB
submissions/          ~258 MB
.venv-metal/          ~2 GB
```

These paths are intentionally ignored by `.gitignore`.

## Recommended Options

### Option 1: Recreate data with Kaggle API

Commit only code and documentation. Each user downloads data locally:

```bash
mkdir -p Data
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p Data
unzip Data/h-and-m-personalized-fashion-recommendations.zip -d Data
```

This is the cleanest option for GitHub.

### Option 2: External artifact storage

For sharing generated artifacts, use external storage such as:

- Google Drive
- AWS S3
- GCS
- Hugging Face Datasets/Hub
- Kaggle Dataset

Then document download paths here.

### Option 3: Git LFS

Git LFS can store large files, but it is not ideal here because the dataset/artifacts are many GB. GitHub LFS quotas are limited unless upgraded.

If used, track only selected medium-sized artifacts, not the full `Data/images/` folder.

## Current Git Policy

Tracked:

- source code
- block scripts
- docs
- result summaries
- dependency file

Ignored:

- raw Kaggle data
- generated artifacts
- model weights
- submission CSVs
- virtual environment
