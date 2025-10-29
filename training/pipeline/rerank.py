#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path

# Ensure repo root on sys.path
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run(cmd: list):
    print("[RUN]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['yelp', 'amazon'], required=True)
    p.add_argument('--data_cfg', required=True)
    p.add_argument('--rerank_cfg', required=True)
    p.add_argument('--ks', default='5,10,20')
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
    cache_dir = os.path.join(out_root, 'cache', 'item_persona')
    inter_dir = os.path.join(out_root, 'cache', 'user_item')
    rerank_dir = os.path.join(out_root, 'rerank')
    os.makedirs(rerank_dir, exist_ok=True)

    alpha = str(rr_cfg.get('alpha', 0.85))
    user_max_k = str(rr_cfg.get('user_max_k', 6))

    candidates = rp(data_cfg.get('candidates'))
    gt = rp(data_cfg.get('gt'))
    if not candidates or not gt:
        raise ValueError('candidates and gt must be set in data config for rerank/evaluate')

    out_reranked = os.path.join(rerank_dir, 'reranked.jsonl')
    # Minimal internal rerank: copy candidates as reranked, then evaluate.
    import shutil
    shutil.copyfile(candidates, out_reranked)
    eval_script = str(_REPO_ROOT / 'evaluation' / 'eval.py')
    cmd_eval = [
        sys.executable, eval_script,
        '--candidates', out_reranked,
        '--gt', gt,
    ]
    run(cmd_eval)

    print('[OK] Evaluate (reranked=candidates passthrough):', out_reranked)


if __name__ == '__main__':
    main()

