# Persona4Rec

This repository contains the implementation for **"Offline Reasoning for Efficient Recommendation: LLM-Empowered Persona-Profiled Item Indexing"**.

## Overview

Our approach consists of three main components:

1. **Data Preparation**: LLM-based pipeline to extract aspects, generate summaries, and create item personas
2. **Training**: Persona-aware encoder training for user-persona matching
3. **Evaluation**: Comprehensive evaluation with overall and scenario-based metrics

## Repository Structure

```
Persona4Rec/
├── data/                      # Raw and processed datasets
│   ├── raw/                   # Original review and metadata files
│   └── processed/             # Generated personas and training data
├── data_preparation/          # LLM-based data generation pipeline
│   ├── generate.py            # Main LLM orchestrator
│   ├── core.py                # Core utilities
│   ├── {dataset}_tasks.py     # Dataset-specific implementations
│   └── prompts/               # LLM prompts for each task
├── training/                  # Model training and inference
├── evaluation/                # Evaluation scripts and metrics
└── README.md
```

## Quick Start

### Prerequisites

```bash
pip install langchain-openai tqdm 
```

### 1. Data

The `data/` directory contains:

**Raw Data** (`data/raw/{dataset}/`):
- `meta_{dataset}.jsonl` - Item metadata (features, categories, etc.)
- `review_{dataset}.jsonl` - User reviews with ratings and timestamps

**Processed Data** (`data/processed/{dataset}/`):
- `1_aspects.jsonl` - Extracted aspects from reviews
- `2_summary.jsonl` - Item summaries
- `3_personas.jsonl` - Generated item personas (5 per item)
- `3.5_history_loo.jsonl` - Leave-One-Out user histories
- `4_selected_personas.jsonl` - Persona-user matching results

See [`data/README.md`](data/README.md) for detailed file descriptions.

### 2. Data Preparation

Generate item personas and user profiles from raw review data using LLMs.

#### Configuration

```bash
cd data_preparation
cp config/profile_info.json.template config/profile_info.json
# Edit profile_info.json to add your OpenAI API key
```

#### Pipeline Execution

Edit `run.sh` to set `DATASET` and `TASK`, then run:

```bash
# Stage 1: Extract aspects from reviews
DATASET=amazon TASK=extract_aspects bash run.sh

# Stage 2: Generate item summaries
DATASET=amazon TASK=generate_summary bash run.sh

# Stage 3: Generate item personas
DATASET=amazon TASK=generate_personas bash run.sh

# Stage 4: Prepare intermediate data (no LLM calls)
DATASET=amazon TASK=prepare_intermediate bash run.sh

# Stage 5: Match personas to users (LLM as Judge)
DATASET=amazon TASK=select_personas bash run.sh
```

**Available Tasks:**
- `extract_aspects` - Extract key aspects from user reviews
- `generate_summary` - Summarize items from metadata
- `generate_personas` - Create 5 diverse personas per item
- `prepare_intermediate` - Build histories and ground truth (no API cost)
- `select_personas` - Match user preferences to personas

See [`data_preparation/README.md`](data_preparation/README.md) for advanced usage and customization.

### 3. Training

Train persona-aware encoder models for user-persona matching and recommendation. Please create new env and install requirements.txt

#### Configuration
```bash
cd training
# Edit data paths in configs/{dataset}/data.yaml
# Adjust hyperparameters in configs/{dataset}/train.yaml
```

#### Pipeline Execution

The training pipeline consists of three stages:
```bash
# Stage 1: Train encoder
python3 training/pipeline/train_encoder.py \
  --dataset amazon \
  --data_cfg training/configs/amazon/data.yaml \
  --train_cfg training/configs/amazon/train.yaml

# Stage 2: Build caches (user-item interaction, item-persona)
python3 training/pipeline/build_cache.py \
  --dataset amazon \
  --data_cfg training/configs/amazon/data.yaml \
  --rerank_cfg training/configs/amazon/rerank.yaml

# Stage 3: Rerank candidates and evaluate
python3 training/pipeline/rerank.py \
  --dataset amazon \
  --data_cfg training/configs/amazon/data.yaml \
  --rerank_cfg training/configs/amazon/rerank.yaml \
  --ks 5,10,20
```

**Pipeline Stages:**
- `train_encoder.py` - Train user-profile encoder with contrastive loss
- `build_cache.py` - Generate embeddings for users and personas
- `rerank.py` - Rerank candidates using cached embeddings

See [`training/README.md`](training/README.md) for detailed configuration and advanced usage.

### 4. Evaluation

Evaluate recommendation performance with comprehensive metrics.

#### Basic Usage

```bash
cd evaluation

# Overall metrics (HIT@K, MRR@K, NDCG@K for K=5,10,20)
python eval.py \
    --candidates <candidate.jsonl> \
    --gt <gt.jsonl>

# With scenario analysis (warm/cold users, head/tail items)
python eval.py \
    --candidates <candidate.jsonl> \
    --gt <gt.jsonl> \
    --review <review.jsonl> \
    --scenarios
```

#### Examples

```bash
# Amazon Dataset
python eval.py \
    --candidates example/amazon/candidate/candidate.jsonl \
    --gt example/amazon/gt/gt.jsonl \
    --review ../data/raw/review_amazon.jsonl \
    --scenarios

# Yelp Dataset
python eval.py \
    --candidates example/yelp/candidate/candidate.jsonl \
    --gt example/yelp/gt/gt.jsonl \
    --review ../data/raw/review_yelp.jsonl \
    --scenarios
```

**Metrics:**
- **HIT@K** (HR@K): Hit rate
- **MRR@K**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain

**Scenario Analysis:** Warm/Cold users, Head/Tail items

See [`evaluation/README.md`](evaluation/README.md) for detailed usage and data formats.
