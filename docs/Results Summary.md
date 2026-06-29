# Results Summary

Validation metrics from the final hybrid ranking/reranking pipeline:

| Metric | Value |
|---|---:|
| MAP@12 | 0.05041 |
| HitRate@12 | 0.21515 |
| Recall@12 | 0.09476 |
| MRR@12 | 0.10988 |
| Catalog coverage | 0.08508 |
| Personalization | 0.91815 |
| Repeat item share | 0.23511 |
| New item share 30d | 0.32348 |
| New item share 90d | 0.45015 |
| Average item popularity percentile | 0.91759 |

Candidate/ranker training scale:

| Artifact | Count |
|---|---:|
| PyTorch retrieval candidates | 1,000,000 |
| Heuristic candidates | 10,278,401 |
| Unique hybrid candidates | 11,188,097 |
| Ranker rows | 11,188,097 |
| Positive rate | 0.001295 |

Final submission was generated locally at `submissions/submission.csv` and passed format validation:

- `1,371,980` rows
- columns: `customer_id`, `prediction`
- exactly 12 article ids per prediction
- customer ids match `sample_submission.csv` order
