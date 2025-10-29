"""
Amazon Dataset Task Implementations

Implements dataset-specific data preparation and post-processing for Amazon Books.

Key Features:
- Extracts review aspects from book reviews (rating, title, text)
- Generates summaries from book metadata (title, author, features, description)
- Prepares persona generation inputs from grouped aspects and summaries
- Implements LLM-as-Judge persona selection logic

Data Schema:
- Reviews: parent_asin (item_id), user_id, rating, timestamp, review_title, review_text
- Metadata: parent_asin, title, author, categories, features, description
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from core import load_jsonl, save_jsonl, parse_json_output, group_aspects


# ============================================================================
# Task 1: Extract Aspects (Amazon Reviews)
# ============================================================================

def prepare_aspects_input(prompt: str, input_path: str, num_sample: Optional[int] = None) -> List[Dict]:
    """Prepare input for Amazon review aspect extraction"""
    print("Loading reviews for aspect extraction...")
    
    all_data = []
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if num_sample and idx >= num_sample:
                break
            
            line = line.strip()
            if not line:
                continue
            
            review = json.loads(line)
            
            all_data.append({
                'review_id': idx,
                'item_id': review['parent_asin'],
                'user_id': review['user_id'],
                'rating': review.get('rating', 0),
                'timestamp': review.get('timestamp', 0),
                'review_title': review.get('title', ''),
                'review_text': review.get('text', ''),
                'model_input': prompt.format(review=review.get('text', ''))
            })
    
    print(f"  Loaded {len(all_data)} reviews")
    return all_data


def postprocess_aspects(result: Dict) -> Dict:
    """Post-process Amazon aspect extraction results"""
    parsed = parse_json_output(result['output'])
    
    # Return only necessary fields (exclude model_input and output)
    return {
        'review_id': result.get('review_id'),
        'item_id': result.get('item_id'),
        'user_id': result.get('user_id'),
        'rating': result.get('rating'),
        'timestamp': result.get('timestamp'),
        'review_title': result.get('review_title'),
        'review_text': result.get('review_text'),
        'extracted_aspects': parsed if parsed else {}
    }


# ============================================================================
# Task 2: Generate Summary (Amazon Books)
# ============================================================================

def prepare_summary_input(prompt: str, input_path: str, num_sample: Optional[int] = None) -> List[Dict]:
    """Prepare input for Amazon book summary generation"""
    print("Loading item metadata for summary generation...")
    
    all_data = []
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if num_sample and idx >= num_sample:
                break
            
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            
            # Extract metadata
            title = item.get('title', '')
            author_info = item.get('author', {})
            author_name = author_info.get('name', '') if isinstance(author_info, dict) else ''
            categories = item.get('categories', [])
            features = item.get('features', [])
            description = item.get('description', [])
            
            # Combine features and description
            full_description = " ".join(features + description)
            
            # Build prompt
            model_input = prompt.format(
                title=title,
                author_name=author_name,
                categories=" > ".join(categories) if categories else "N/A",
                description=full_description
            )
            
            all_data.append({
                'summary_id': idx,
                'item_id': item['parent_asin'],
                'item_title': title,
                'item_description': full_description,
                'item_categories': categories,
                'model_input': model_input
            })
    
    print(f"  Loaded {len(all_data)} items")
    return all_data


def postprocess_summary(result: Dict) -> Dict:
    """Post-process Amazon summary generation results"""
    # Return only necessary fields (exclude model_input and output)
    return {
        'item_id': result.get('item_id'),
        'item_title': result.get('item_title'),
        'item_description': result.get('item_description'),
        'item_summary': result['output']
    }


# ============================================================================
# Task 3: Generate Personas (Amazon)
# ============================================================================

def prepare_personas_input(prompt: str, aspects_path: str, summary_path: str) -> List[Dict]:
    """Prepare input for Amazon persona generation"""
    print("Preparing persona generation input...")
    
    # Check if grouped file exists
    grouped_path = aspects_path.replace("1_aspects.jsonl", "2.5_grouped.jsonl")
    
    if os.path.exists(grouped_path):
        print(f"  Using existing grouped file: {grouped_path}")
        grouped_data = load_jsonl(grouped_path)
    else:
        print(f"  Grouped file not found. Creating: {grouped_path}")
        grouped_data = group_aspects(aspects_path, summary_path)
        save_jsonl(grouped_data, grouped_path)
        print(f"  Saved grouped file: {grouped_path}")
    
    # Build prompts
    all_data = []
    for item in grouped_data:
        # Format review extractions
        review_extractions = []
        for i, aspect in enumerate(item['aspects'], 1):
            extracted = aspect.get('extracted_aspects', {})
            if not extracted:
                continue
            
            # Clean extracted (remove null values)
            clean_extracted = {}
            for key, value in extracted.items():
                if value and str(value).strip() and str(value).lower() not in ['null', 'none', 'n/a']:
                    clean_extracted[key] = value
            
            if clean_extracted:
                review_extractions.append(f"Review {i} Extracted Aspects:")
                review_extractions.append(json.dumps(clean_extracted, ensure_ascii=False, indent=2))
                review_extractions.append("")
        
        review_text = "\n".join(review_extractions)
        
        model_input = prompt.format(
            item_summary=item.get('item_summary', 'No summary available'),
            review_extractions=review_text
        )
        
        all_data.append({
            'item_id': item['item_id'],
            'item_title': item.get('item_title', ''),
            'item_description': item.get('item_description', ''),
            'item_summary': item.get('item_summary', ''),
            'review_count': item.get('aspect_count', 0),
            'model_input': model_input
        })
    
    print(f"  Prepared {len(all_data)} items")
    return all_data


def postprocess_personas(result: Dict) -> Dict:
    """Post-process Amazon persona generation results"""
    parsed = parse_json_output(result['output'])
    personas = parsed.get('personas', []) if parsed else []
    
    # Return only necessary fields (exclude model_input and output)
    return {
        'item_id': result.get('item_id'),
        'item_title': result.get('item_title'),
        'item_description': result.get('item_description'),
        'item_summary': result.get('item_summary'),
        'review_count': result.get('review_count'),
        'personas': personas,
        'persona_count': len(personas)
    }


# ============================================================================
# Task 4: Select Personas (Amazon)
# ============================================================================

def prepare_selection_input(
    prompt: str,
    history_loo_path: str,
    gt_path: str,
    persona_path: str,
    min_history: int = 6,
    num_users: Optional[int] = None
) -> List[Dict]:
    """Prepare input for Amazon persona selection"""
    print("Preparing persona selection input...")
    
    # Load data
    history_loo_data = load_jsonl(history_loo_path)
    gt_data = {gt['user_id']: gt['item_id'] for gt in load_jsonl(gt_path)}
    personas_data = load_jsonl(persona_path)
    
    # Index personas by item_id (통일된 필드 사용)
    personas_by_item = {}
    for p_record in personas_data:
        item_id = p_record.get('item_id')
        if item_id:
            personas_by_item[item_id] = p_record.get('personas', [])
    
    print(f"  Loaded {len(history_loo_data)} users with LOO history")
    print(f"  Loaded {len(gt_data)} GT items")
    print(f"  Loaded {len(personas_by_item)} items with personas")
    
    # Prepare selection tasks
    all_data = []
    processed_users = 0
    skipped_no_gt = 0
    skipped_no_persona = 0
    skipped_insufficient_history = 0
    
    for user_record in history_loo_data:
        if num_users and processed_users >= num_users:
            break
        
        user_id = user_record['user_id']
        
        # Check if user has GT
        if user_id not in gt_data:
            skipped_no_gt += 1
            continue
        
        target_item_id = gt_data[user_id]
        
        # Check if target has personas
        if target_item_id not in personas_by_item:
            skipped_no_persona += 1
            continue
        
        personas = personas_by_item[target_item_id]
        if not personas:
            skipped_no_persona += 1
            continue
        
        # Get LOO history (already sorted by timestamp)
        loo_reviews = user_record.get('reviews', [])
        
        # Filter: positive (rating >= 3) + negative (rating <= 2)
        positive_reviews = [r for r in loo_reviews if r.get('rating', 0) >= 3]
        negative_reviews = [r for r in loo_reviews if r.get('rating', 0) <= 2]
        
        # Combine and take most recent 6
        combined_reviews = positive_reviews + negative_reviews
        combined_reviews = sorted(combined_reviews, key=lambda x: x.get('timestamp', 0), reverse=True)[:6]
        
        # Check minimum history requirement
        if len(combined_reviews) < min_history:
            skipped_insufficient_history += 1
            continue
        
        # Build user profile text
        profile_parts = []
        for r in combined_reviews:
            sentiment = "Liked" if r.get('rating', 0) >= 3 else "Disliked"
            review_title = r.get('review_title', 'Untitled')
            summary = r.get('summary', 'No summary')
            profile_parts.append(f"- {sentiment}: {review_title} | {summary}")
        user_profile = "\n".join(profile_parts)
        
        # Build persona options text
        persona_options_parts = []
        for p in personas:
            persona_options_parts.append(
                f"{p['ID']}. {p['Name']}: {p['Description']} "
                f"(Reasoning: {p.get('Preference_Rationale', 'No reasoning')})"
            )
        persona_options = "\n".join(persona_options_parts)
        
        # Build target item text (would need to get from persona record, but we don't have it here)
        target_item = f"Target Item ID: {target_item_id}"
        
        # Build model input
        model_input = prompt.format(
            user_profile=user_profile,
            target_item=target_item,
            persona_options=persona_options
        )
        
        all_data.append({
            'user_id': user_id,
            'target_item_id': target_item_id,
            'user_history': combined_reviews,
            'personas': personas,
            'model_input': model_input
        })
        
        processed_users += 1
    
    print(f"  Prepared {len(all_data)} users for persona selection")
    if skipped_no_gt:
        print(f"  Skipped {skipped_no_gt} users (no GT)")
    if skipped_no_persona:
        print(f"  Skipped {skipped_no_persona} users (no personas for target)")
    if skipped_insufficient_history:
        print(f"  Skipped {skipped_insufficient_history} users (insufficient history < {min_history})")
    
    return all_data


def postprocess_selection(result: Dict, personas: List[Dict]) -> Dict:
    """Post-process Amazon persona selection results - parse LLM response"""
    import re
    
    output = result.get('output', '')
    
    # Parse: "Selection: P1" or "Selection: P3"
    match = re.search(r'Selection:\s*(P\d+)', output, re.IGNORECASE)
    
    selected_persona_id = None
    selected_persona_name = None
    selected_persona = None
    
    if match:
        selected_id = match.group(1).upper()  # Ensure uppercase (P1, not p1)
        
        # Find the persona by ID
        for p in personas:
            if p.get('ID') == selected_id:
                selected_persona = p
                break
        
        selected_persona_id = selected_id
        selected_persona_name = selected_persona.get('Name', '') if selected_persona else ''
    
    # Extract reasoning
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\nSelection:|$)', output, re.IGNORECASE | re.DOTALL)
    selection_reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    
    # Return only necessary fields (exclude model_input and output)
    return {
        'user_id': result.get('user_id'),
        'target_item_id': result.get('target_item_id'),
        'selected_persona_id': selected_persona_id,
        'selected_persona_name': selected_persona_name,
        'selected_persona': selected_persona,
        'selection_reasoning': selection_reasoning
    }
