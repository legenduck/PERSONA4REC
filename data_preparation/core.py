"""
Core Utilities for LLM-based Data Generation

Provides common functionality for the persona generation pipeline:
- File I/O operations (JSONL format)
- OpenAI API configuration and async batch processing
- Data grouping and parsing utilities

This module is dataset-agnostic and shared across all implementations.
"""

import json
import os
import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Optional
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from tqdm.asyncio import tqdm_asyncio


# ============================================================================
# File I/O
# ============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data as JSONL"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_prompt(file_path: str) -> str:
    """Load prompt template from file"""
    with open(file_path, 'r') as f:
        return f.read().strip()


# ============================================================================
# OpenAI API
# ============================================================================

def set_openai_api(api_owner: str) -> str:
    """Set OpenAI API key from config and return it"""
    config_path = os.path.join(os.path.dirname(__file__), "config/profile_info.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if api_owner not in config:
        raise ValueError(f"API owner '{api_owner}' not found in config")
    
    api_key = config[api_owner]["api_key"]
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


# ============================================================================
# JSON Parsing
# ============================================================================

def parse_json_output(text: str) -> Optional[Dict]:
    """Parse JSON from LLM output, handling code blocks"""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                lines = lines[:i]
                break
        text = '\n'.join(lines).strip()
    
    # Try to extract JSON object
    if '{' in text:
        start = text.find('{')
        end = text.rfind('}') + 1
        text = text[start:end]
    
    try:
        return json.loads(text)
    except:
        return None


# ============================================================================
# Grouping Logic (Common for aspects + summaries → personas)
# ============================================================================

def group_aspects(aspects_path: str, summary_path: str) -> List[Dict]:
    """
    Group extracted aspects by item_id and join with item summaries.
    
    This function prepares input for persona generation by combining:
    - Multiple aspect extractions per item
    - Item summary information
    
    Args:
        aspects_path: Path to aspects JSONL file (Stage 1 output)
        summary_path: Path to summaries JSONL file (Stage 2 output)
        
    Returns:
        List of dictionaries with item metadata, aspects array, and aspect count
    """
    print("Grouping aspects by item_id...")
    
    # Load and group aspects
    grouped = defaultdict(list)
    with open(aspects_path, 'r') as f:
        for line in f:
            if line.strip():
                aspect = json.loads(line)
                grouped[aspect['item_id']].append(aspect)
    
    # Load summaries
    summaries = {}
    with open(summary_path, 'r') as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                summaries[s['item_id']] = s
    
    # Join
    results = []
    for item_id, aspects in grouped.items():
        summary = summaries.get(item_id, {})
        results.append({
            'item_id': item_id,
            'item_title': summary.get('item_title', ''),
            'item_description': summary.get('item_description', ''),
            'item_summary': summary.get('item_summary', ''),
            'aspects': aspects,
            'aspect_count': len(aspects)
        })
    
    print(f"  Grouped into {len(results)} items")
    return results


# ============================================================================
# Async LangChain Batch Processing
# ============================================================================

TOTAL_COST = 0.0

async def async_generate(llm: ChatOpenAI, model_data: Dict, idx: int) -> Dict:
    """
    Execute a single asynchronous OpenAI API call with cost tracking.
    
    Args:
        llm: LangChain ChatOpenAI instance
        model_data: Dictionary containing 'model_input' and other metadata
        idx: Index for error reporting
        
    Returns:
        Dictionary with original data plus 'output' field containing LLM response
    """
    global TOTAL_COST
    
    system_message = SystemMessage(content=model_data['model_input'])
    
    while True:
        try:
            response = await llm.agenerate([[system_message]])
            
            # Calculate cost
            input_tokens = response.llm_output['token_usage']['prompt_tokens']
            output_tokens = response.llm_output['token_usage']['completion_tokens']
            
            if llm.model_name == "gpt-4o-mini":
                TOTAL_COST += input_tokens / 1_000_000 * 0.150
                TOTAL_COST += output_tokens / 1_000_000 * 0.600
            elif llm.model_name == "gpt-4o":
                TOTAL_COST += input_tokens / 1_000_000 * 5.00
                TOTAL_COST += output_tokens / 1_000_000 * 15.00
            
            break
        except Exception as e:
            print(f"\n[ERROR] {idx}: {e}")
            await asyncio.sleep(1)
            continue
    
    result = deepcopy(model_data)
    result['output'] = response.generations[0][0].text
    
    return result


async def generate_concurrently(all_model_data: List[Dict], start_idx: int, 
                                model_name: str, temperature: float, max_tokens: int) -> List[Dict]:
    """
    Execute batch of LLM calls concurrently with progress tracking.
    
    Args:
        all_model_data: List of dictionaries containing 'model_input' and metadata
        start_idx: Starting index for progress reporting
        model_name: OpenAI model name (e.g., 'gpt-4o-mini')
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        List of dictionaries with original data plus 'output' field
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=50
    )
    
    tasks = [async_generate(llm, model_data, i + start_idx)
             for i, model_data in enumerate(all_model_data)]
    
    return await tqdm_asyncio.gather(*tasks)
