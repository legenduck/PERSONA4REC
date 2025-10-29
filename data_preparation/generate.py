#!/usr/bin/env python3
"""
LLM-based Data Generation Pipeline

This script orchestrates persona-based data generation using OpenAI's API.
Supports multiple datasets (Amazon, Yelp) with dataset-specific processing.

Pipeline Stages:
    1. extract_aspects: Extract user profile aspects from reviews
    2. generate_summary: Generate item summaries from metadata
    3. generate_personas: Create diverse user personas for items
    4. select_personas: Match user profiles to personas (LLM as Judge)

Usage:
    python generate.py --task <task_name> --dataset <dataset> [options]
    
Example:
    python generate.py --task extract_aspects --dataset amazon \\
        --input_path ../data/raw/review_amazon.jsonl \\
        --prompt_path prompts/amazon/aspects.txt \\
        --save_dir ../data/processed/amazon/
"""

import os
import sys
import argparse
import asyncio
from typing import List, Dict

# Import core utilities
import core
from core import (
    set_openai_api,
    load_prompt,
    generate_concurrently,
    save_jsonl
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based Data Generation Pipeline"
    )
    
    # Basic arguments
    parser.add_argument("--api_owner", default="default", 
                       help="API owner key in config/profile_info.json")
    parser.add_argument("--task", 
                       choices=["extract_aspects", "generate_summary", 
                               "generate_personas", "select_personas"],
                       required=True,
                       help="Which task to run")
    parser.add_argument("--dataset", type=str, default="amazon", 
                       choices=["amazon", "yelp"],
                       help="Dataset name (amazon, yelp)")
    
    # Input/output paths
    parser.add_argument("--input_path", type=str, 
                       help="Input data file path (for tasks 1-2)")
    parser.add_argument("--aspects_path", type=str, 
                       help="Aspect results path (for task 3)")
    parser.add_argument("--summary_path", type=str, 
                       help="Summary results path (for task 3)")
    parser.add_argument("--prompt_path", type=str, required=True, 
                       help="Prompt file path")
    parser.add_argument("--save_dir", type=str, required=True, 
                       help="Output directory")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", 
                       help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=500, 
                       help="Maximum tokens to generate")
    
    # Sampling
    parser.add_argument("--num_sample", type=int, default=None, 
                       help="Number of samples to process (for testing)")
    parser.add_argument("--num_users", type=int, default=None, 
                       help="Number of users for selection task")
    parser.add_argument("--min_history", type=int, default=6, 
                       help="Minimum history length for selection task")
    
    return parser.parse_args()


def load_and_prepare_data(args, tasks_module):
    """Load prompt and prepare data based on task"""
    prompt = load_prompt(args.prompt_path)
    print(f"Loaded prompt from: {args.prompt_path}")
    
    if args.task == "extract_aspects":
        return tasks_module.prepare_aspects_input(
            prompt, args.input_path, args.num_sample
        )
    
    elif args.task == "generate_summary":
        return tasks_module.prepare_summary_input(
            prompt, args.input_path, args.num_sample
        )
    
    elif args.task == "generate_personas":
        return tasks_module.prepare_personas_input(
            prompt, args.aspects_path, args.summary_path
        )
    
    elif args.task == "select_personas":
        # Use 3.5 intermediate files
        base_dir = f"../data/processed/{args.dataset}"
        history_loo_path = f"{base_dir}/3.5_history_loo.jsonl"
        gt_path = f"{base_dir}/3.5_gt.jsonl"
        persona_path = f"{base_dir}/3.5_persona_with_summary.jsonl"
        
        return tasks_module.prepare_selection_input(
            prompt,
            history_loo_path,
            gt_path,
            persona_path,
            min_history=args.min_history,
            num_users=args.num_users
        )
    
    else:
        raise ValueError(f"Unknown task: {args.task}")


async def main_async(args):
    """Main async function"""
    
    # Import dataset-specific task module
    if args.dataset == "amazon":
        import amazon_tasks as tasks
    elif args.dataset == "yelp":
        import yelp_tasks as tasks
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print("=" * 80)
    print()
    
    # Set API key
    api_key = set_openai_api(args.api_owner)
    print(f"API key loaded for: {args.api_owner}\n")
    
    # Load and prepare data
    print("Preparing data...")
    prepared_data = load_and_prepare_data(args, tasks)
    print(f"Total items to process: {len(prepared_data)}\n")
    
    if len(prepared_data) == 0:
        print("No data to process. Exiting.")
        return
    
    # Run async generation
    print("Starting LLM generation...")
    all_results = await generate_concurrently(
        all_model_data=prepared_data,
        start_idx=0,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Post-process results
    print("\nPost-processing results...")
    if args.task == "extract_aspects":
        all_results = [tasks.postprocess_aspects(r) for r in all_results]
    
    elif args.task == "generate_summary":
        all_results = [tasks.postprocess_summary(r) for r in all_results]
    
    elif args.task == "generate_personas":
        all_results = [tasks.postprocess_personas(r) for r in all_results]
    
    elif args.task == "select_personas":
        all_results = [
            tasks.postprocess_selection(r, r['personas']) 
            for r in all_results
        ]
    
    # Save results
    print("\nSaving results...")
    
    # Determine output file name based on task
    task_to_filename = {
        "extract_aspects": "1_aspects.jsonl",
        "generate_summary": "2_summary.jsonl",
        "generate_personas": "3_personas.jsonl",
        "select_personas": "4_selected_personas.jsonl"
    }
    
    output_file = os.path.join(args.save_dir, task_to_filename[args.task])
    save_jsonl(all_results, output_file)
    
    print(f"  Saved to: {output_file}")
    print(f"  Total items: {len(all_results)}")
    print(f"  Total cost: ${core.TOTAL_COST:.4f}")
    print("\nDone!")


def main():
    """Entry point"""
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

