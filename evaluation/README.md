# Evaluation

Unified evaluation script for recommendation performance analysis across different datasets.

## Script

### `eval.py`

Dataset-agnostic evaluation script supporting both overall and scenario-based analysis.
Works with any dataset using standardized `user_id` and `item_id` fields.

**Metrics:**
- **HIT@K** (HR@K): Hit rate (equals Recall@K in Leave-One-Out setting)
- **MRR@K**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain

**Default K values:** 5, 10, 20

**Scenarios (with `--scenarios` flag, K=10 only):**
- **Warm Users@10**: High-activity users 
- **Cold Users@10**: Low-activity users 
- **Head Items@10**: Popular items
- **Tail Items@10**: Long-tail items 

Note: Scenario analysis only reports HR@10 and NDCG@10

## Usage

### Overall Evaluation

```bash
python eval.py --candidates <candidate.jsonl> --gt <gt.jsonl>
```

### Scenario Analysis

```bash
python eval.py --candidates <candidate.jsonl> --gt <gt.jsonl> \
               --review <review.jsonl> --scenarios
```

### Custom K values

```bash
# For overall evaluation only (scenario analysis always uses K=10)
python eval.py --candidates <candidate.jsonl> --gt <gt.jsonl> --ks 1,5,10,20,50
```

## Examples

### Amazon Books

```bash
# Overall evaluation
python eval.py \
    --candidates example/amazon/candidate/candidate.jsonl \
    --gt example/amazon/gt/gt.jsonl

# With scenario analysis
python eval.py \
    --candidates example/amazon/candidate/candidate.jsonl \
    --gt example/amazon/gt/gt.jsonl \
    --review ../data/raw/review_amazon.jsonl \
    --scenarios
```

### Yelp Restaurants

```bash
# Overall evaluation
python eval.py \
    --candidates example/yelp/candidate/candidate.jsonl \
    --gt example/yelp/gt/gt.jsonl

# With scenario analysis
python eval.py \
    --candidates example/yelp/candidate/candidate.jsonl \
    --gt example/yelp/gt/gt.jsonl \
    --review ../data/raw/review_yelp.jsonl \
    --scenarios
```

## Data Format

All datasets use standardized field names:

**Candidate JSONL:**
```json
{
  "user_id": "USER123",
  "results": [
    {"item_id": "ITEM456", "score": 9.23},
    {"item_id": "ITEM789", "score": 8.57}
  ]
}
```

**Ground Truth JSONL:**
```json
{"user_id": "USER123", "item_id": "ITEM456"}
```

**Review JSONL (for scenario analysis):**
```json
{
  "user_id": "USER123",
  "item_id": "ITEM456",
  "rating": 5.0,
  "timestamp": 1234567890
}
```

## Output Format

**Overall Performance:**
```
HIT@5  MRR@5  NDCG@5  HIT@10  MRR@10  NDCG@10  HIT@20  MRR@20  NDCG@20
0.0707 0.0385 0.0464  0.1081  0.0434  0.0585   0.1446  0.0460  0.0679
```

**Scenario Analysis (K=10):**
```
       Warm Users@10         Cold Users@10         Head Items@10         Tail Items@10
     HR@10   NDCG@10       HR@10   NDCG@10       HR@10   NDCG@10       HR@10   NDCG@10
    0.0599    0.0320      0.1432    0.0789      0.1707    0.0904      0.0360    0.0196
```

