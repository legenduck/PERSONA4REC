#!/usr/bin/env python3
"""
Intermediate Data Preparation (Stage 3.5)

Prepares structured data files for the persona selection stage without LLM calls.
This script performs data transformations and joins to create:
- User interaction history (chronologically sorted)
- Ground truth (GT) for evaluation
- Leave-One-Out (LOO) history for temporal evaluation
- Persona catalog with item summaries

Usage:
    python prepare_intermediate_data.py --dataset <dataset>
    
Example:
    python prepare_intermediate_data.py --dataset amazon
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} records to: {file_path}")


def build_user_history(aspects_path, summary_path):
    """
    Build chronologically sorted user interaction history.
    
    Combines extracted aspects with item summaries to create complete
    user history records for each user. Each record contains review information,
    item summary, and temporal metadata.
    
    Args:
        aspects_path: Path to aspects file (1_aspects.jsonl)
        summary_path: Path to summaries file (2_summary.jsonl)
        
    Returns:
        List of user history records with reviews sorted by timestamp
    """
    print("Building user history...")
    
    # Load data
    aspects = load_jsonl(aspects_path)
    summaries = {s['item_id']: s for s in load_jsonl(summary_path)}
    
    # Group by user
    user_reviews = defaultdict(list)
    
    for aspect in aspects:
        user_id = aspect['user_id']
        item_id = aspect['item_id']
        
        # Get summary for this item
        summary_data = summaries.get(item_id, {})
        
        review = {
            'user_id': user_id,
            'item_id': item_id,
            'review_title': aspect.get('review_title', ''),
            'summary': summary_data.get('item_summary', ''),
            'extracted': aspect.get('extracted_aspects', {}),
            'timestamp': aspect.get('timestamp', 0),
            'rating': aspect.get('rating', 0.0)
        }
        
        user_reviews[user_id].append(review)
    
    # Sort by timestamp and create final structure
    user_history = []
    for user_id, reviews in user_reviews.items():
        sorted_reviews = sorted(reviews, key=lambda x: x['timestamp'])
        user_history.append({
            'user_id': user_id,
            'reviews': sorted_reviews
        })
    
    print(f"  Created history for {len(user_history)} users")
    return user_history


def extract_ground_truth(history_data, min_rating=3.0):
    """
    Extract ground truth items for evaluation (Leave-One-Out strategy).
    
    For each user, identifies the most recent (last) positive interaction
    (rating >= min_rating) as the ground truth item to predict.
    
    Args:
        history_data: User history from build_user_history()
        min_rating: Minimum rating threshold for positive interactions (default: 3.0)
        
    Returns:
        List of ground truth records with user_id and target item_id
    """
    print(f"Extracting ground truth (rating >= {min_rating}, last interaction)...")
    
    gt_data = []
    
    for user_record in history_data:
        user_id = user_record['user_id']
        reviews = user_record['reviews']
        
        # Filter by rating
        positive_reviews = [r for r in reviews if r.get('rating', 0) >= min_rating]
        
        if positive_reviews:
            # Get the last (most recent) positive interaction
            gt_item = positive_reviews[-1]
            gt_data.append({
                'user_id': user_id,
                'item_id': gt_item['item_id']
            })
    
    print(f"  Found GT for {len(gt_data)}/{len(history_data)} users")
    return gt_data


def build_loo_history(history_data, gt_data):
    """
    Build Leave-One-Out history: remove GT item from each user's history
    """
    print("Building LOO history...")
    
    # Create GT lookup
    gt_lookup = {gt['user_id']: gt['item_id'] for gt in gt_data}
    
    loo_history = []
    
    for user_record in history_data:
        user_id = user_record['user_id']
        reviews = user_record['reviews']
        
        # Skip user if no GT
        if user_id not in gt_lookup:
            continue
        
        gt_item_id = gt_lookup[user_id]
        
        # Remove GT item
        loo_reviews = [r for r in reviews if r['item_id'] != gt_item_id]
        
        # Only keep users with remaining history
        if loo_reviews:
            loo_history.append({
                'user_id': user_id,
                'reviews': loo_reviews
            })
    
    print(f"  Created LOO history for {len(loo_history)} users")
    return loo_history


def inject_summary_to_personas(persona_path, summary_path):
    """
    Inject item summaries into persona catalog
    """
    print("Injecting summaries into personas...")
    
    # Load data
    personas_data = load_jsonl(persona_path)
    summaries = {s['item_id']: s['item_summary'] for s in load_jsonl(summary_path)}
    
    persona_catalog = []
    
    for persona_record in personas_data:
        item_id = persona_record['item_id']
        item_summary = summaries.get(item_id, '')
        
        # Add summary to each persona
        personas_with_summary = []
        for persona in persona_record.get('personas', []):
            persona_copy = persona.copy()
            persona_copy['Item_Summary'] = item_summary
            persona_copy['unique_persona_id'] = f"{item_id}_{persona['ID']}"
            personas_with_summary.append(persona_copy)
        
        persona_catalog.append({
            'item_id': item_id,
            'personas': personas_with_summary
        })
    
    print(f"  Created persona catalog for {len(persona_catalog)} items")
    return persona_catalog


def main():
    parser = argparse.ArgumentParser(description='Prepare intermediate data files')
    parser.add_argument('--task', type=str, required=True,
                       choices=['build_history', 'extract_gt', 'build_loo', 'inject_summary', 'all'],
                       help='Task to run')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # Set paths
    base_dir = f"{args.data_dir}/{args.dataset}"
    aspects_path = f"{base_dir}/1_aspects.jsonl"
    summary_path = f"{base_dir}/2_summary.jsonl"
    persona_path = f"{base_dir}/3_personas.jsonl"
    
    history_path = f"{base_dir}/3.5_history.jsonl"
    gt_path = f"{base_dir}/3.5_gt.jsonl"
    loo_path = f"{base_dir}/3.5_history_loo.jsonl"
    persona_summary_path = f"{base_dir}/3.5_persona_with_summary.jsonl"
    
    print("=" * 80)
    print(f"Prepare Intermediate Data - Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)
    print()
    
    if args.task in ['build_history', 'all']:
        print("[1/4] Building user history...")
        history_data = build_user_history(aspects_path, summary_path)
        save_jsonl(history_data, history_path)
        print()
    
    if args.task in ['extract_gt', 'all']:
        print("[2/4] Extracting ground truth...")
        if args.task == 'extract_gt':
            history_data = load_jsonl(history_path)
        gt_data = extract_ground_truth(history_data)
        save_jsonl(gt_data, gt_path)
        print()
    
    if args.task in ['build_loo', 'all']:
        print("[3/4] Building LOO history...")
        if args.task == 'build_loo':
            history_data = load_jsonl(history_path)
            gt_data = load_jsonl(gt_path)
        loo_data = build_loo_history(history_data, gt_data)
        save_jsonl(loo_data, loo_path)
        print()
    
    if args.task in ['inject_summary', 'all']:
        print("[4/4] Injecting summaries into personas...")
        persona_catalog = inject_summary_to_personas(persona_path, summary_path)
        save_jsonl(persona_catalog, persona_summary_path)
        print()
    
    print("=" * 80)
    print("Intermediate data preparation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

