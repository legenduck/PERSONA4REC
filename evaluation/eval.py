#!/usr/bin/env python3
"""
Unified Evaluation Script for Recommendation Systems

Evaluates candidate rankings against ground truth using standard metrics.
Supports overall evaluation and scenario-based analysis (warm/cold users, head/tail items).
Works with any dataset using standardized user_id and item_id fields.

Output Format:
    Overall: HIT@5 MRR@5 NDCG@5 HIT@10 MRR@10 NDCG@10 HIT@20 MRR@20 NDCG@20
    Scenarios (K=10 only): HR@10 and NDCG@10 for Warm/Cold/Head/Tail

Usage:
    # Overall evaluation (K=5,10,20 by default)
    python eval.py --candidates candidate.jsonl --gt gt.jsonl
    
    # With scenario analysis (requires review data, K=10 only)
    python eval.py --candidates candidate.jsonl --gt gt.jsonl \\
                   --review reviews.jsonl --scenarios
    
    # Custom K values (for overall only)
    python eval.py --candidates candidate.jsonl --gt gt.jsonl --ks 1,5,10,20,50
"""

import argparse
import json
import sys
import math
from typing import Dict, Set, Tuple
from collections import defaultdict


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_gt(path: str) -> Dict[str, Set[str]]:
    """
    Load ground truth from JSONL file.
    
    Args:
        path: Path to GT JSONL file with user_id and item_id fields
        
    Returns:
        Dictionary mapping user_id to set of ground truth item_ids
    """
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user_id = obj.get('user_id')
            item_id = obj.get('item_id')
            if user_id and item_id:
                gt.setdefault(user_id, set()).add(item_id)
    return gt


# ============================================================================
# Evaluation Metrics (Leave-One-Out)
# ============================================================================

def evaluate_overall(candidates_path: str, gt: Dict[str, Set[str]], ks=(5, 10, 20)) -> Dict[int, Dict[str, float]]:
    """
    Evaluate overall performance across all users.
    
    Metrics:
        - HIT@K: Hit rate (equals Recall@K in LOO setting)
        - MRR@K: Mean Reciprocal Rank
        - NDCG@K: Normalized Discounted Cumulative Gain
    
    Args:
        candidates_path: Path to candidate JSONL file
        gt: Ground truth dictionary
        ks: Tuple of K values for top-K evaluation
        
    Returns:
        Dictionary with metrics per K value
    """
    sums = {k: {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0} for k in ks}
    num_users = 0

    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            user_id = obj.get('user_id')
            if user_id not in gt:
                continue
            
            # Extract ranked item list
            results = obj.get('results', [])
            items = [r.get('item_id') for r in results if r and r.get('item_id')]
            if not items:
                continue
            
            num_users += 1
            true_items = gt[user_id]
            
            # Find first relevant item rank (LOO assumes single positive per user)
            first_rank = None
            for idx, item_id in enumerate(items, start=1):
                if item_id in true_items:
                    first_rank = idx
                    break
            
            # Calculate metrics for each K
            for k in ks:
                if first_rank and first_rank <= k:
                    sums[k]['HIT'] += 1.0
                    sums[k]['MRR'] += 1.0 / first_rank
                    sums[k]['NDCG'] += 1.0 / math.log2(first_rank + 1.0)
    
    # Average over users
    results = {}
    for k in ks:
        if num_users > 0:
            results[k] = {
                'HIT': sums[k]['HIT'] / num_users,
                'MRR': sums[k]['MRR'] / num_users,
                'NDCG': sums[k]['NDCG'] / num_users,
                'users': num_users,
            }
        else:
            results[k] = {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0, 'users': 0}
    
    return results


# ============================================================================
# Scenario Analysis (Warm/Cold Users, Head/Tail Items)
# ============================================================================

def extract_candidate_users(candidates_path: str) -> Set[str]:
    """Extract user IDs from candidate file."""
    users = set()
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user_id = obj.get('user_id')
            if user_id:
                users.add(user_id)
    return users


def detect_dataset_type(review_path: str) -> str:
    """
    Detect dataset type from review data format.
    
    Args:
        review_path: Path to review JSONL file
        
    Returns:
        Dataset type: 'amazon' or 'yelp'
        
    Raises:
        ValueError: If dataset format is not supported
    """
    with open(review_path, 'r', encoding='utf-8') as f:
        # Check first few lines to determine format
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 lines
                break
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            
            if 'parent_asin' in obj and 'user_id' in obj:
                return 'amazon'
            elif 'business_id' in obj and 'user_id' in obj:
                return 'yelp'
    
    raise ValueError("Unsupported dataset format. Expected 'parent_asin + user_id' (Amazon) or 'business_id + user_id' (Yelp)")


def load_statistics(review_path: str, target_users: Set[str], target_items: Set[str]) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """
    Load interaction counts from review data with dataset detection.
    
    Args:
        review_path: Path to review JSONL file
        target_users: Set of users to count
        target_items: Set of items to count
        
    Returns:
        Tuple of (item_counts, user_counts, dataset_type)
        
    Raises:
        ValueError: If dataset format is not supported
    """
    # Detect dataset type
    dataset_type = detect_dataset_type(review_path)
    
    item_counts = {}
    user_counts = {}
    
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            # Extract fields based on dataset type
            if dataset_type == 'amazon':
                item_id = obj.get('parent_asin')
            elif dataset_type == 'yelp':
                item_id = obj.get('business_id')
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            user_id = obj.get('user_id')
            
            if item_id and item_id in target_items:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1
            if user_id and user_id in target_users:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1
    
    return item_counts, user_counts, dataset_type


def classify_by_percentile(counts: Dict[str, int], top_p=0.8, bottom_p=0.2) -> Tuple[Set[str], Set[str]]:
    """
    Classify entities into head/tail (or warm/cold) by percentile.
    
    Args:
        counts: Dictionary of entity counts
        top_p: Percentile threshold for head (default: 0.8, meaning top 20%)
        bottom_p: Percentile threshold for tail (default: 0.2, meaning bottom 20%)
        
    Returns:
        Tuple of (head_set, tail_set)
    """
    if not counts:
        return set(), set()
    
    sorted_counts = sorted(counts.values())
    n = len(sorted_counts)
    
    top_idx = max(0, int(n * top_p) - 1)
    bottom_idx = min(n - 1, int(n * bottom_p))
    
    top_threshold = sorted_counts[top_idx]
    bottom_threshold = sorted_counts[bottom_idx]
    
    head = {k for k, v in counts.items() if v >= top_threshold}
    tail = {k for k, v in counts.items() if v <= bottom_threshold}
    
    return head, tail


def evaluate_scenarios(
    candidates_path: str,
    gt: Dict[str, Set[str]],
    head_items: Set[str],
    tail_items: Set[str],
    warm_users: Set[str],
    cold_users: Set[str],
    ks=(5, 10, 20)
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Evaluate performance across different scenarios.
    
    Scenarios:
        - warm: High-activity users (top 20%)
        - cold: Low-activity users (bottom 20%)
        - head: Popular items (top 20%)
        - tail: Long-tail items (bottom 20%)
    
    Args:
        candidates_path: Path to candidate JSONL file
        gt: Ground truth dictionary
        head_items: Set of popular items
        tail_items: Set of long-tail items
        warm_users: Set of high-activity users
        cold_users: Set of low-activity users
        ks: Tuple of K values
        
    Returns:
        Dictionary with metrics per K value and scenario
    """
    segments = {
        'warm': {'users': set(), 'sums': {k: {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0} for k in ks}},
        'cold': {'users': set(), 'sums': {k: {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0} for k in ks}},
        'head': {'users': set(), 'sums': {k: {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0} for k in ks}},
        'tail': {'users': set(), 'sums': {k: {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0} for k in ks}},
    }

    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            user_id = obj.get('user_id')
            if user_id not in gt:
                continue
            
            results = obj.get('results', [])
            items = [r.get('item_id') for r in results if r and r.get('item_id')]
            if not items:
                continue
            
            true_items = gt[user_id]
            
            # Determine scenarios for this user
            user_gt_is_head = any(item in head_items for item in true_items)
            user_gt_is_tail = any(item in tail_items for item in true_items)
            
            scenarios = []
            if user_id in warm_users:
                scenarios.append('warm')
            if user_id in cold_users:
                scenarios.append('cold')
            if user_gt_is_head:
                scenarios.append('head')
            if user_gt_is_tail:
                scenarios.append('tail')
            
            # Find first relevant item rank
            first_rank = None
            for idx, item_id in enumerate(items, start=1):
                if item_id in true_items:
                    first_rank = idx
                    break
            
            # Update metrics for applicable scenarios
            for scenario in scenarios:
                segments[scenario]['users'].add(user_id)
                
                for k in ks:
                    if first_rank and first_rank <= k:
                        segments[scenario]['sums'][k]['HIT'] += 1.0
                        segments[scenario]['sums'][k]['MRR'] += 1.0 / first_rank
                        segments[scenario]['sums'][k]['NDCG'] += 1.0 / math.log2(first_rank + 1.0)
    
    # Average over users per scenario
    results = {}
    for k in ks:
        results[k] = {}
        for scenario, data in segments.items():
            num_users = len(data['users'])
            if num_users > 0:
                results[k][scenario] = {
                    'HIT': data['sums'][k]['HIT'] / num_users,
                    'MRR': data['sums'][k]['MRR'] / num_users,
                    'NDCG': data['sums'][k]['NDCG'] / num_users,
                    'users': num_users,
                }
            else:
                results[k][scenario] = {'HIT': 0.0, 'MRR': 0.0, 'NDCG': 0.0, 'users': 0}
    
    return results


# ============================================================================
# Output Formatting
# ============================================================================

def print_table(results: Dict, scenario_mode: bool = False, dataset_type: str = None):
    """
    Print evaluation results in table format.
    
    Args:
        results: Evaluation results dictionary
        scenario_mode: If True, print scenario breakdown (K=10 only)
        dataset_type: Dataset type for display (amazon/yelp)
    """
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    if dataset_type:
        print(f"Dataset: {dataset_type.capitalize()}")
    print("=" * 100)
    
    if scenario_mode:
        # Scenario mode: Only K=10, HR and NDCG only, horizontal layout
        k = 10
        if k not in results:
            print(f"Warning: K={k} not found in results for scenario analysis.")
            return
        
        print("\nScenario Analysis (K=10)")
        print("-" * 100)
        
        # Scenario order
        scenario_keys = ['warm', 'cold', 'head', 'tail']
        scenario_labels = {
            'warm': 'Warm Users@10',
            'cold': 'Cold Users@10',
            'head': 'Head Items@10',
            'tail': 'Tail Items@10'
        }
        
        # Build header: scenario names
        header_parts = []
        for key in scenario_keys:
            if key in results[k]:
                header_parts.append(f"{scenario_labels[key]:>20}")
        header = "  ".join(header_parts)
        print(header)
        
        # Build metric row: HR NDCG for each scenario
        metric_parts = []
        for key in scenario_keys:
            if key in results[k]:
                metric_parts.append(f"{'HR@10':>10} {'NDCG@10':>9}")
        metrics_header = "  ".join(metric_parts)
        print(metrics_header)
        print("-" * 100)
        
        # Build values
        value_parts = []
        for key in scenario_keys:
            if key in results[k]:
                metrics = results[k][key]
                value_parts.append(f"{metrics['HIT']:>10.4f} {metrics['NDCG']:>9.4f}")
        values = "  ".join(value_parts)
        print(values)
    else:
        # Overall mode: All K values in one line
        ks = sorted(results.keys())
        
        # Build header
        header_parts = []
        for k in ks:
            header_parts.extend([f"HIT@{k}", f"MRR@{k}", f"NDCG@{k}"])
        
        print("\nOverall Performance")
        print("-" * 100)
        header = "  ".join(f"{h:>10}" for h in header_parts)
        print(header)
        print("-" * 100)
        
        # Build values
        value_parts = []
        for k in ks:
            metrics = results[k]
            value_parts.extend([
                f"{metrics['HIT']:>10.4f}",
                f"{metrics['MRR']:>10.4f}",
                f"{metrics['NDCG']:>10.4f}"
            ])
        
        values = "  ".join(value_parts)
        print(values)
        
        # Print user count
        print("-" * 100)
        print(f"Total Users: {results[ks[0]]['users']}")
    
    print("=" * 100 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation candidates against ground truth"
    )
    parser.add_argument('--candidates', required=True, 
                       help='Path to candidate JSONL file')
    parser.add_argument('--gt', required=True,
                       help='Path to ground truth JSONL file')
    parser.add_argument('--scenarios', action='store_true',
                       help='Enable scenario analysis (warm/cold/head/tail)')
    parser.add_argument('--review', default=None,
                       help='Path to review JSONL file (required for scenario analysis)')
    parser.add_argument('--ks', default='5,10,20',
                       help='Comma-separated K values (default: 5,10,20)')
    args = parser.parse_args()
    
    # Parse K values
    try:
        ks = tuple(int(x.strip()) for x in args.ks.split(',') if x.strip())
        if not ks:
            ks = (5, 10, 20)
    except Exception:
        print("Error: Invalid --ks format. Using default (5,10,20).", file=sys.stderr)
        ks = (5, 10, 20)
    
    # Load ground truth
    print("Loading ground truth...", file=sys.stderr)
    gt = load_gt(args.gt)
    print(f"Loaded GT for {len(gt)} users", file=sys.stderr)
    
    if args.scenarios:
        # Scenario analysis mode
        if not args.review:
            print("Error: --review is required for scenario analysis", file=sys.stderr)
            sys.exit(1)
        
        print("\nExtracting candidate users...", file=sys.stderr)
        candidate_users = extract_candidate_users(args.candidates)
        print(f"Found {len(candidate_users)} users in candidates", file=sys.stderr)
        
        # Extract GT items for candidate users
        gt_items = set()
        for user_id in candidate_users:
            if user_id in gt:
                gt_items.update(gt[user_id])
        print(f"Found {len(gt_items)} unique GT items", file=sys.stderr)
        
        # Load statistics from review data
        print("\nLoading review statistics...", file=sys.stderr)
        item_counts, user_counts, dataset_type = load_statistics(args.review, candidate_users, gt_items)
        
        print(f"Detected dataset: {dataset_type.capitalize()}", file=sys.stderr)
        
        # Classify into scenarios
        head_items, tail_items = classify_by_percentile(item_counts, top_p=0.8, bottom_p=0.2)
        warm_users, cold_users = classify_by_percentile(user_counts, top_p=0.8, bottom_p=0.2)
        
        print(f"Head items: {len(head_items)}, Tail items: {len(tail_items)}", file=sys.stderr)
        print(f"Warm users: {len(warm_users)}, Cold users: {len(cold_users)}", file=sys.stderr)
        
        # Evaluate scenarios
        print("\nEvaluating scenarios...", file=sys.stderr)
        results = evaluate_scenarios(
            args.candidates, gt, head_items, tail_items, warm_users, cold_users, ks=ks
        )
        print_table(results, scenario_mode=True, dataset_type=dataset_type)
    else:
        # Overall evaluation only
        print("\nEvaluating overall performance...", file=sys.stderr)
        results = evaluate_overall(args.candidates, gt, ks=ks)
        print_table(results, scenario_mode=False)


if __name__ == '__main__':
    main()

