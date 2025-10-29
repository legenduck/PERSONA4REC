#!/bin/bash
set -e

# ============================================================================
# Persona-based Recommendation Data Generation Pipeline
# 
# This script executes different stages of the data generation pipeline.
# Edit DATASET and TASK variables below to run specific stages.
# ============================================================================

# Dataset selection: amazon | yelp
DATASET=yelp

# Task selection: extract_aspects | generate_summary | generate_personas | prepare_intermediate | select_personas
TASK=select_personas

# ============================================================================
# Task Configuration
# ============================================================================

if [ "$TASK" = "extract_aspects" ]; then
    # Stage 1: Extract user profile aspects from reviews
    python generate.py \
        --task extract_aspects \
        --dataset ${DATASET} \
        --input_path ../data/raw/review_${DATASET}.jsonl \
        --prompt_path prompts/${DATASET}/aspects.txt \
        --save_dir ../data/processed/${DATASET} \
        --model_name gpt-4o-mini \
        --temperature 0.0 \
        --max_tokens 500 \
        #--num_sample 100

elif [ "$TASK" = "generate_summary" ]; then
    # Stage 2: Generate item summaries from metadata
    python generate.py \
        --task generate_summary \
        --dataset ${DATASET} \
        --input_path ../data/raw/meta_${DATASET}.jsonl \
        --prompt_path prompts/${DATASET}/summary.txt \
        --save_dir ../data/processed/${DATASET} \
        --model_name gpt-4o-mini \
        --temperature 0.3 \
        --max_tokens 300 \
        #--num_sample 100

elif [ "$TASK" = "generate_personas" ]; then
    # Stage 3: Generate diverse user personas for each item
    python generate.py \
        --task generate_personas \
        --dataset ${DATASET} \
        --aspects_path ../data/processed/${DATASET}/1_aspects.jsonl \
        --summary_path ../data/processed/${DATASET}/2_summary.jsonl \
        --prompt_path prompts/${DATASET}/persona.txt \
        --save_dir ../data/processed/${DATASET} \
        --model_name gpt-4o-mini \
        --temperature 0.0 \
        --max_tokens 2000

elif [ "$TASK" = "prepare_intermediate" ]; then
    # Stage 3.5: Prepare intermediate data (no LLM calls)
    # Creates: history, GT, LOO history, persona catalog
    echo "Preparing intermediate data (3.5 stage)..."
    python prepare_intermediate_data.py \
        --task all \
        --dataset ${DATASET} \
        --data_dir ../data/processed

elif [ "$TASK" = "select_personas" ]; then
    # Stage 4: Select matching personas (LLM as Judge)
    python generate.py \
        --task select_personas \
        --dataset ${DATASET} \
        --prompt_path prompts/${DATASET}/selection.txt \
        --save_dir ../data/processed/${DATASET} \
        --model_name gpt-4o-mini \
        --temperature 0.1 \
        --max_tokens 300 \
        #--num_users 10 \

else
    echo "Error: Invalid TASK. Must be one of: extract_aspects, generate_summary, generate_personas, prepare_intermediate, select_personas"
    exit 1
fi

