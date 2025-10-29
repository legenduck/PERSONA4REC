# Training

End-to-end training pipeline for building user-profile encoders and preparing caches for reranking and evaluation.

## Directory Structure
```
training/
├── configs/
│   ├── amazon/
│   │   ├── data.yaml          # Data paths (selected, persona, history, optional items, candidates/gt)
│   │   ├── train.yaml         # Training hyperparameters
│   │   └── rerank.yaml        # Cache/rerank settings
│   └── yelp/
│       ├── data.yaml
│       ├── train.yaml
│       └── rerank.yaml
├── models/
│   ├── datasets/
│   │   ├── amazon.py          # Amazon Dataset loader (strict schema)
│   │   └── yelp.py            # Yelp Dataset loader (strict schema)
│   ├── encoder.py             # Sentence-transformer based encoder wrapper
│   └── trainer.py             # Training loop and contrastive loss
└── pipeline/
    ├── train_encoder.py       # Train encoder (Yelp/Amazon)
    ├── build_cache.py         # Build interaction/persona caches (orchestrator)
    ├── rerank.py              # Rerank candidates and evaluate
    └── cache_builders/        # Internal cache building modules
        ├── interaction_amazon.py  # Amazon user-item interaction cache
        ├── interaction_yelp.py    # Yelp user-item interaction cache
        ├── persona_amazon.py      # Amazon item-persona cache
        └── persona_yelp.py        # Yelp item-persona cache
```
## Data Configs
Edit the dataset config before running:
- `selected`: 4_selected_personas.jsonl
- `persona`: 3.5_persona_with_summary.jsonl (or your persona file)
- `loo_history`: 3.5_history_loo.jsonl
- Optional: `item_summary` (2_summary.jsonl), `aspects` (2.5_grouped.jsonl)
- Rerank only: `candidates`, `gt`
- Output root: `out_root`

Example (Amazon):
```yaml
selected: ../../data/processed/amazon/4_selected_personas.jsonl
persona: ../../data/processed/amazon/3.5_persona_with_summary.jsonl
loo_history: ../../data/processed/amazon/3.5_history_loo.jsonl
item_summary: ../../data/processed/amazon/2_summary.jsonl
aspects: ../../data/processed/amazon/2.5_grouped.jsonl
candidates: ../../evaluation/example/amazon/candidate/candidate.jsonl
gt: ../../evaluation/example/amazon/gt/gt.jsonl
out_root: ./outputs/amazon
```

## Usage

### 1) Train Encoder
```bash
python3 training/pipeline/train_encoder.py \
  --dataset amazon \ # yelp|amazon
  --data_cfg training/configs/amazon/data.yaml \
  --train_cfg training/configs/amazon/train.yaml
```
Notes:
- Uses AMP by default. Set `USE_AMP=0` to disable.
- Multi-GPU: launch with torchrun and set WORLD_SIZE/LOCAL_RANK.

### 2) Build Caches
Build user-item interaction cache and item-persona cache using the trained checkpoint.
```bash
python3 training/pipeline/build_cache.py \
  --dataset amazon \
  --data_cfg training/configs/amazon/data.yaml \
  --rerank_cfg training/configs/amazon/rerank.yaml \
  --model_dir BAAI/bge-m3  # Optional: defaults to trained model in out_root/train
```
Notes:
- Internally uses `cache_builders/` modules for memmap-based efficient caching
- Supports multi-GPU parallel processing
- `rerank.yaml` fields:
  - `mode`: summary | aspects (formerly 'component')
  - `fields`: persona fields used for persona cache (e.g., "N,D,I")
  - `num_gpus`: Number of GPUs for parallel cache building

### 3) Rerank & Evaluate
```bash
python3 training/pipeline/rerank.py \
  --dataset amazon \
  --data_cfg training/configs/amazon/data.yaml \
  --rerank_cfg training/configs/amazon/rerank.yaml \
  --ks 5,10,20
```
Requirements:
- `data.yaml` must include `candidates` and `gt` paths.
- Reranked results and metrics are saved under `out_root`.

## Schema Notes
- Strict keys expected by loaders:
  - Personas: `unique_persona_id`, `Name`, `Description`, `Item_Summary`, `Preference_Rationale`
  - History: `user_id`, `item_id` (Yelp: business_id), `review_title`, `summary`, `extracted`, `rating`, `timestamp`
  - Selection: `target_item_id`, `selected_persona_id`
- Text builders include persona `Preference_Rationale` and interaction component fields.

## Tips
- Set `max_steps` in `train.yaml` for quick smoke tests.
- Ensure `out_root` has write permission.
- For aspect mode, set `mode: aspects` and provide `aspects` path in data config.






