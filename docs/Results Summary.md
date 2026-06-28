# Results Summary

Validation metrics from the final hybrid ranking/reranking pipeline:

| Metric | Value |
|---|---:|
| MAP@12 | 0.05017 |
| HitRate@12 | 0.21395 |
| Recall@12 | 0.09467 |
| MRR@12 | 0.10907 |
| Catalog coverage | 0.08759 |
| Personalization | 0.91677 |
| Repeat item share | 0.23805 |
| New item share 30d | 0.32128 |
| New item share 90d | 0.44923 |
| Average item popularity percentile | 0.91728 |

Candidate/ranker training scale:

| Artifact | Count |
|---|---:|
| PyTorch retrieval candidates | 1,000,000 |
| Heuristic candidates | 10,278,401 |
| Unique hybrid candidates | 11,183,222 |
| Ranker rows | 11,183,222 |
| Positive rate | 0.001296 |

Final submission was generated locally at `submissions/submission.csv` and passed format validation:

- `1,371,980` rows
- columns: `customer_id`, `prediction`
- exactly 12 article ids per prediction
- customer ids match `sample_submission.csv` order
