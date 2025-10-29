#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
from pathlib import Path

# Ensure repo root on sys.path
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import cache builders
from training.pipeline.cache_builders import interaction_yelp, interaction_amazon
from training.pipeline.cache_builders import persona_yelp, persona_amazon


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['yelp', 'amazon'], required=True)
    p.add_argument('--data_cfg', required=True)
    p.add_argument('--rerank_cfg', required=True)
    p.add_argument('--model_dir', default=None, help='Override encoder directory (defaults to out_root/train)')
    args = p.parse_args()

    with open(args.data_cfg, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    with open(args.rerank_cfg, 'r', encoding='utf-8') as f:
        rr_cfg = yaml.safe_load(f)

    cfg_base = Path(args.data_cfg).resolve().parent

    # Resolve paths relative to data config location when not absolute
    def rp(p: str) -> str:
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        return str((cfg_base / p).resolve())

    out_root = str((_REPO_ROOT / (data_cfg.get('out_root') or f"training/outputs/{args.dataset}")).resolve())
    train_out = os.path.join(out_root, 'train')
    model_dir = rp(args.model_dir) if args.model_dir else train_out
    inter_out = os.path.join(out_root, 'cache', 'user_item')
    per_out = os.path.join(out_root, 'cache', 'item_persona')
    os.makedirs(inter_out, exist_ok=True)
    os.makedirs(per_out, exist_ok=True)

    # Normalize mode: accept 'aspects' (preferred) and legacy 'component' as the same
    raw_mode = str(rr_cfg.get('mode', 'summary')).lower()
    mode = 'aspects' if raw_mode in ('aspects', 'component') else 'summary'
    num_gpus = int(rr_cfg.get('num_gpus', 1))
    fields = str(rr_cfg.get('fields', 'N,D,I,R'))

    # ---------------- Build caches using cache_builders modules -----------------
    print(f"[INFO] Building caches for {args.dataset} dataset")
    
    history = rp(data_cfg['loo_history'])
    item_summary = rp(data_cfg.get('item_summary', ''))
    aspects_path = rp(data_cfg.get('aspects', ''))
    persona_path = rp(data_cfg.get('persona'))
    
    if args.dataset == 'yelp':
        # Build Yelp interaction cache
        print(f"[INFO] Building Yelp interaction cache -> {inter_out}")
        interaction_yelp.main_with_args(
            history=history,
            item_summary=item_summary,
            aspects=aspects_path if mode == 'aspects' else '',
            mode=mode,
            model=model_dir,
            out_dir=inter_out,
            batch_size=128,
            max_len=512,
            num_gpus=num_gpus
        )
        
        # Build Yelp persona cache
        print(f"[INFO] Building Yelp persona cache -> {per_out}")
        persona_yelp.main_with_args(
            model=model_dir,
            out_dir=per_out,
            persona_path=persona_path,
            fields=fields,
            batch_size=512,
            persona_max_len=512
        )
    else:  # amazon
        # Build Amazon interaction cache
        print(f"[INFO] Building Amazon interaction cache -> {inter_out}")
        interaction_amazon.main_with_args(
            history=history,
            item_summary=item_summary,
            aspects=aspects_path if mode == 'aspects' else '',
            mode=mode,
            model=model_dir,
            out_dir=inter_out,
            batch_size=128,
            max_len=512,
            num_gpus=num_gpus
        )
        
        # Build Amazon persona cache
        print(f"[INFO] Building Amazon persona cache -> {per_out}")
        persona_amazon.main_with_args(
            model=model_dir,
            out_dir=per_out,
            persona_path=persona_path,
            fields=fields,
            batch_size=1024,
            persona_max_len=512
        )

    print(f"[OK] Caches built successfully:")
    print(f"  - Interaction cache: {inter_out}")
    print(f"  - Persona cache: {per_out}")


if __name__ == '__main__':
    main()

