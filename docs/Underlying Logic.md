# H&M Recommender Capstone: Thought Process

## Project Goal

The goal is to build an end-to-end recommender system for the H&M Personalized Fashion Recommendations task. The system should recommend 12 articles for each customer based on historical purchase behavior.

Rather than optimizing only for a leaderboard score, we focused on building a production-style recommender pipeline:

```text
Data processing
-> Candidate retrieval
-> Ranking
-> Reranking
-> Evaluation
-> Submission generation
```

This mirrors how large-scale recommender systems are usually built. Retrieval finds a manageable set of plausible items. Ranking orders those items with richer features. Reranking applies final list-level cleanup before serving.

## Block 1: Data Processing

We started by making the time structure explicit. Since recommendations are about future purchases, random splits would leak future behavior. We used chronological splits:

- Training data: transactions up to `2020-08-23`
- Validation window: `2020-08-24` to `2020-09-06`
- Test/deployment-style window: after `2020-09-06`

From this, we created:

- customer id mappings
- article id mappings
- padded user purchase histories
- next-item training examples
- validation/test labels
- popularity fallback items

The main training signal for retrieval is next-item prediction: given a customer's recent purchase history, predict the next article they bought.

This gave us about 21.4M retrieval training examples.

## Block 2: Item Content Features

Fashion recommendations depend heavily on item appearance. The dataset includes product images, so we used precomputed MobileNet image embeddings aligned to our internal article ids.

The image embedding artifact has:

- shape: `(100758, 576)`
- about `99.56%` image coverage
- a zero vector for the padding item

These embeddings let the model represent items not only by id, but also by visual similarity. This is useful for fashion because two articles may be semantically similar even if they have different ids.

## Block 3: DNN Retrieval

For retrieval, we built a PyTorch two-tower model.

The user tower encodes a customer's recent purchase history. The item tower encodes candidate articles using item id embeddings and image embeddings. Both towers output vectors in the same embedding space.

Training objective:

```text
user history vector should be close to the next purchased item vector
and far from other items in the batch
```

We used in-batch negatives for efficiency. Each batch contains many positive user-item pairs, and the other items in the same batch serve as negative examples.

The model learned a meaningful signal:

- loss decreased from about `6.31` to `5.39`
- in-batch Recall@10 rose to about `0.21`

However, retrieval-only MAP@12 was `0.00156`. That told us the neural retrieval model was learning, but pure DNN retrieval was not enough for final recommendation quality.

## Why Retrieval Alone Was Not Enough

Retrieval is optimized for broad candidate coverage, not final precision. The H&M task is also strongly affected by short-term and business-specific signals:

- recent popularity
- repeat purchases
- category affinity
- age-group trends
- co-purchase patterns

A two-tower model can learn some of these indirectly, but not as reliably as explicit candidate sources and ranker features.

This led to the key design decision: keep the DNN retrieval model, but make it one candidate source inside a hybrid candidate generation system.

## Hybrid Candidate Generation

We expanded candidate retrieval with multiple sources:

- **DNN retrieval candidates:** learned semantic/user-history similarity
- **Repeat purchases:** items the customer recently bought
- **Recent global popularity:** items trending in the latest time window
- **Age-bin popularity:** popular items among customers in similar age groups
- **Category popularity:** popular articles in categories the user recently interacted with
- **Co-occurrence candidates:** items bought together or near each other by other users

This is a production-style design. The DNN captures latent similarity, while heuristic generators inject domain and business knowledge.

The ranker receives source features such as:

```text
src_torch_retrieval
src_repeat
src_global_pop
src_age_pop
src_category_pop
src_cooc
retrieval_rank
source_score
```

This lets the ranker learn how much to trust each source in context.

## Block 5: Ranking

After candidate generation, we trained a LightGBM LambdaRank model. The ranker receives the hybrid candidates and learns to order them based on validation purchases.

Feature groups include:

- candidate source flags
- retrieval rank and source score
- item popularity
- user activity level
- user-item recency
- item metadata
- customer age/activity features
- price gap between user preference and item average price

The ranker is the stage that turns broad candidate coverage into precise recommendations.

In the full run, the hybrid candidate set contained:

- PyTorch candidates: `1,000,000`
- heuristic candidates: `10,278,401`
- unique hybrid candidates: `11,183,222`
- positive rate: `0.001296`

## Block 6: Reranking

The final reranking step produces the top 12 recommendations per customer.

It sorts candidates by ranker score and applies light list-level cleanup, including mild category diversity. This reflects the idea that final recommendation quality is not only item-by-item relevance, but also the composition of the final list.

## Current Result

The final validation result is:

```text
MAP@12 = 0.05017
HitRate@12 = 0.21395
Recall@12 = 0.09467
MRR@12 = 0.10907
```

This is a large improvement over pure DNN retrieval:

```text
DNN retrieval only:        0.00156
Hybrid ranking/reranking:  0.05017
```

The result confirms that the system architecture is working. The DNN retrieval model contributes learned candidates, while the hybrid retrieval and ranking stages add the short-term and business-specific signals that matter strongly in fashion recommendations.

We also added list-quality and business metrics:

```text
catalog_coverage = 0.08759
personalization = 0.91677
repeat_item_share = 0.23805
new_item_share_30d = 0.32128
new_item_share_90d = 0.44923
avg_item_popularity_percentile = 0.91728
```

These metrics make the project more production-oriented. MAP measures relevance, while the additional metrics describe discovery, personalization, popularity bias, repeat behavior, and list diversity.

## Final Architecture

```text
Raw transactions, articles, customers, images
        |
        v
Block 1: time split, id mapping, histories, labels
        |
        v
Block 2: item image embeddings
        |
        v
Block 3: PyTorch two-tower DNN retrieval
        |
        v
Block 4: retrieval candidate evaluation
        |
        +-----------------------------+
        |                             |
        v                             v
DNN candidates              Business/domain candidates
                             - repeat purchase
                             - recent popularity
                             - age-bin popularity
                             - category popularity
                             - co-occurrence
        |                             |
        +-------------+---------------+
                      |
                      v
Block 5: LightGBM ranking
                      |
                      v
Block 6: reranking and MAP@12 evaluation
                      |
                      v
Block 7: Kaggle submission generation
```

## Main Lesson

The most important lesson is that a practical recommender is rarely a single model. The DNN retrieval model is valuable, but it works best as part of a system.

The final design combines:

- learned representation retrieval
- domain-specific candidate generation
- supervised ranking
- final reranking

That is why the project is still DNN-based, but also production-oriented.
