#!/usr/bin/env python3
import argparse
import json
import os
import gzip
import subprocess
from typing import Dict, Any, Iterable, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def read_jsonl_auto(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith('.gz'):
        f = gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    else:
        f = open(path, 'r', encoding='utf-8', errors='ignore')
    try:
        for ln in f:
            s = ln.strip()
            if s:
                yield json.loads(s)
    finally:
        f.close()


def build_text_summary(r: Dict[str, Any], bid2summary: Dict[str, str]) -> str:
    parts: List[str] = ['query: [UserProfile-Interaction]']
    bid = r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin')
    if isinstance(bid, str) and bid.strip():
        parts.append(f"Business ID: {bid.strip()}")
    summ = bid2summary.get(str(bid)) or ''
    if isinstance(summ, str) and summ.strip():
        parts.append(f"Item Summary: {summ.strip()}")
    return '\n'.join(parts)


def _stringify_component_value(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        try:
            return ', '.join([str(x).strip() for x in val if str(x).strip()])
        except Exception:
            return ''
    try:
        s = str(val)
        return s.strip()
    except Exception:
        return ''


def build_text_component(r: Dict[str, Any], comp_map: Dict[Tuple[str, str], Dict[str, Any]], bid2summary: Dict[str, str], user_id: str) -> str:
    parts: List[str] = ['query: [UserProfile-Interaction]']
    bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
    summ = (bid2summary.get(bid) or '').strip()
    if summ:
        parts.append(f"Item Summary: {summ}")
    ext = comp_map.get((bid, str(user_id))) or {}
    if isinstance(ext, dict):
        for k, v in ext.items():
            vs = _stringify_component_value(v)
            if vs:
                parts.append(f"{k}: {vs}")
    return '\n'.join(parts)


def build_text_review(bid: str, summary_text: str, extracted: Dict[str, Any]) -> str:
    parts: List[str] = ['query: [UserProfile-Interaction]']
    if isinstance(bid, str) and bid.strip():
        parts.append(f"Business ID: {bid.strip()}")
    if isinstance(summary_text, str) and summary_text.strip():
        parts.append(f"Item Summary: {summary_text.strip()}")
    if isinstance(extracted, dict) and extracted:
        for k, v in extracted.items():
            vs = _stringify_component_value(v)
            if vs:
                parts.append(f"{k}: {vs}")
    return '\n'.join(parts)


# --- Sharding helpers and merge (defined before main so they are available) ---
def _hash_to_shard(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    import hashlib
    h = hashlib.sha1(key.encode('utf-8')).digest()
    val = int.from_bytes(h[:8], byteorder='big', signed=False)
    return val % max(1, num_shards)


def user_belongs_to_shard(user_id: str, shard_idx: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return _hash_to_shard(user_id, num_shards) == int(shard_idx)


def merge_shard_outputs(shard_dirs: List[str], out_dir: str) -> None:
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    shard_dirs = [os.path.abspath(p) for p in shard_dirs]
    shard_dirs = sorted(shard_dirs)
    metas = []
    uids_lists: List[List[str]] = []
    sizes: List[int] = []
    dim = None
    for d in shard_dirs:
        meta_path = os.path.join(d, 'cache_meta.json')
        uids_path = os.path.join(d, 'uids.txt')
        mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        if not (os.path.exists(meta_path) and os.path.exists(uids_path) and os.path.exists(mmap_path)):
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(uids_path, 'r', encoding='utf-8') as f:
            uids = [ln.strip() for ln in f if ln.strip()]
        if dim is None:
            dim = int(meta['dim'])
        metas.append(meta)
        uids_lists.append(uids)
        sizes.append(len(uids))
    total = sum(sizes)
    if dim is None:
        raise ValueError('No shard metas found to merge')
    out_mmap_path = os.path.join(out_dir, 'interaction_embeds.fp16.mmap')
    out_mmap = np.memmap(out_mmap_path, mode='w+', dtype=np.float16, shape=(total, int(dim)))
    offset = 0
    for d, n in zip(shard_dirs, sizes):
        shard_mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        shard = np.memmap(shard_mmap_path, mode='r', dtype=np.float16, shape=(n, int(dim)))
        out_mmap[offset:offset+n, :] = shard[:]
        offset += n
    out_mmap.flush()
    # Merge uids
    with open(os.path.join(out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for uids in uids_lists:
            for u in uids:
                f.write(u + '\n')
    # Merge index.json
    merged_index: Dict[str, List[str]] = {}
    for d in shard_dirs:
        idx_path = os.path.join(d, 'index.json')
        if os.path.exists(idx_path):
            with open(idx_path, 'r', encoding='utf-8') as f:
                part_idx = json.load(f)
            for uid, lst in part_idx.items():
                if not isinstance(lst, list):
                    continue
                merged_index.setdefault(uid, []).extend(lst)
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_index, f)
    # Write meta
    base_meta = metas[0] if metas else {'model': '', 'dim': int(dim), 'normalized': True}
    with open(os.path.join(out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(base_meta, f)


def main_with_args(**kwargs):
    """
    Python API: Build Amazon interaction cache with keyword arguments.
    
    Args:
        history (str): Path to history file
        item_summary (str): Path to item summary file
        aspects (str): Path to aspects file (formerly components)
        mode (str): 'summary' or 'aspects'
        model (str): Model path
        out_dir (str): Output directory
        batch_size (int): Batch size
        max_len (int): Max sequence length
        num_gpus (int): Number of GPUs for parallel processing
        **kwargs: Additional arguments
    """
    # Create a namespace object from kwargs
    class Args:
        pass
    args = Args()
    
    # Set attributes from kwargs
    args.history = kwargs.get('history', '')
    args.item_summary = kwargs.get('item_summary', '')
    args.aspects = kwargs.get('aspects', '')
    args.mode = kwargs.get('mode', 'aspects')
    args.model = kwargs.get('model', '')
    args.out_dir = kwargs.get('out_dir', '')
    args.batch_size = int(kwargs.get('batch_size', 128))
    args.max_len = int(kwargs.get('max_len', 1024))
    args.num_gpus = int(kwargs.get('num_gpus', 1))
    args.min_rating = float(kwargs.get('min_rating', 3.0))
    args.debug_print_inputs = kwargs.get('debug_print_inputs', False)
    args.debug_print_k = int(kwargs.get('debug_print_k', 5))
    args.internal_shard_run = kwargs.get('internal_shard_run', False)
    args.shard_idx = int(kwargs.get('shard_idx', 0))
    args.num_shards = int(kwargs.get('num_shards', 1))
    args.gpu_id = kwargs.get('gpu_id', '')
    
    # Run main logic
    _run_main_logic(args)


def main():
    ap = argparse.ArgumentParser(description='Build Amazon per-interaction embeddings cache (like-only)')
    ap.add_argument('--history', required=True)
    ap.add_argument('--item_summary', required=True)
    ap.add_argument('--aspects', default='', help='Optional components file (jsonl). If present, component fields are included by default')
    ap.add_argument('--mode', default='aspects', choices=['summary','aspects'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--num_gpus', type=int, default=1)
    ap.add_argument('--min_rating', type=float, default=3.0)
    # Debug: print example encoder inputs
    ap.add_argument('--debug_print_inputs', action='store_true', help='Print sample input texts that will be encoded')
    ap.add_argument('--debug_print_k', type=int, default=5, help='How many sample inputs to print')
    # Internal sharding/orchestration
    ap.add_argument('--internal_shard_run', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--shard_idx', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--num_shards', type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument('--gpu_id', default='', help=argparse.SUPPRESS)
    args = ap.parse_args()
    
    _run_main_logic(args)


def _run_main_logic(args):
    """Internal function containing the main logic."""
    os.makedirs(args.out_dir, exist_ok=True)

    # Orchestrate multi-GPU by spawning shard subprocesses, then merge
    if int(args.num_gpus) > 1 and not bool(args.internal_shard_run):
        # Prepare shard dirs
        shard_dirs: List[str] = []
        for i in range(int(args.num_gpus)):
            d = os.path.join(args.out_dir, f'shard_{i}')
            os.makedirs(d, exist_ok=True)
            shard_dirs.append(d)
        procs = []
        this_script = os.path.abspath(__file__)
        for i in range(int(args.num_gpus)):
            # Assign GPU i (recycled if fewer physical GPUs)
            gpu = ''
            try:
                if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
                    # Respect provided mask; child will map index within mask
                    gpu = str(i)
                else:
                    gpu = str(i)
            except Exception:
                gpu = ''
            cmd = [
                'python3', this_script,
                '--history', args.history,
                '--item_summary', args.item_summary,
                '--model', args.model,
                '--out_dir', shard_dirs[i],
                '--batch_size', str(args.batch_size),
                '--max_len', str(args.max_len),
                '--min_rating', str(args.min_rating),
                '--num_gpus', '1',
                '--internal_shard_run',
                '--shard_idx', str(i),
                '--num_shards', str(int(args.num_gpus))
            ]
            if bool(getattr(args, 'debug_print_inputs', False)):
                cmd.append('--debug_print_inputs')
                cmd.extend(['--debug_print_k', str(getattr(args, 'debug_print_k', 5))])
            if gpu != '':
                cmd.extend(['--gpu_id', gpu])
            env = os.environ.copy()
            procs.append(subprocess.Popen(cmd, env=env))
        # Wait
        codes = [p.wait() for p in procs]
        if any(c != 0 for c in codes):
            raise RuntimeError(f'Sharded processes failed with codes: {codes}')
        # Merge
        merge_shard_outputs(shard_dirs, args.out_dir)
        return

    # Load item summaries map
    bid2summary: Dict[str, str] = {}
    for row in read_jsonl_auto(args.item_summary):
        bid = row.get('business_id') or row.get('asin') or row.get('item_id') or row.get('parent_asin')
        if not bid:
            continue
        # Accept multiple possible keys for summary
        bid2summary[str(bid)] = row.get('item_summary') or row.get('Item_summary') or row.get('summary') or ''

    # Prepare model (HuggingFace)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and str(args.gpu_id).strip() != '':
        try:
            torch.cuda.set_device(int(str(args.gpu_id)))
            device = f"cuda:{int(str(args.gpu_id))}"
        except Exception:
            device = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = AutoModel.from_pretrained(args.model, use_safetensors=True, trust_remote_code=True).to(device)
    enc.eval()

    def encode(texts: List[str]) -> np.ndarray:
        batch = tok(texts, padding=True, truncation=True, max_length=int(args.max_len), return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = enc(**batch).last_hidden_state
            attn = batch.get('attention_mask', None)
            if attn is not None:
                attn = attn.unsqueeze(-1).type_as(out)
                emb = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-6)
            else:
                emb = out.mean(dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-6)
        return emb.detach().cpu().to(torch.float16).numpy()

    # Optional: load components (user-specific) map
    comp_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if args.aspects:
        try:
            for row in read_jsonl_auto(args.aspects):
                bid_val = row.get('business_id') or row.get('asin') or row.get('item_id') or row.get('parent_asin')
                if not bid_val:
                    continue
                bid = str(bid_val)
                revs = row.get('reviews')
                if isinstance(revs, list):
                    # Schema A: { reviews: [ { user_id, extracted }, ... ] }
                    for rev in revs:
                        if not isinstance(rev, dict):
                            continue
                        uid_val = rev.get('user_id')
                        ext = rev.get('extracted') or rev.get('components') or {}
                        if not uid_val or not isinstance(ext, dict):
                            continue
                        comp_map[(bid, str(uid_val))] = ext
                else:
                    # Schema B: { review: { user_id, extracted, ... } }
                    rev = row.get('review')
                    if isinstance(rev, dict):
                        uid_val = rev.get('user_id')
                        ext = rev.get('extracted') or rev.get('components') or {}
                        if uid_val and isinstance(ext, dict):
                            comp_map[(bid, str(uid_val))] = ext
        except Exception:
            comp_map = {}

    # Collect like-only interactions
    rows: List[Tuple[str, str, str]] = []  # (user_id, review_uid, text)
    def make_review_uid(uid: str, r: Dict[str, Any]) -> str:
        ts = str(r.get('timestamp') or r.get('date') or '')
        bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
        stars = str(r.get('rating') or r.get('stars') or r.get('score') or '')
        import hashlib
        h = hashlib.sha1(f"{uid}|{bid}|{ts}|{stars}".encode('utf-8')).hexdigest()
        return h[:16]

    user_review_uids: Dict[str, List[str]] = {}
    for row in read_jsonl_auto(args.history):
        uid = str(row.get('user_id') or '')
        # Shard filter: only keep users belonging to this shard when in shard run
        if bool(args.internal_shard_run) and int(args.num_shards) > 1:
            if not user_belongs_to_shard(uid, int(args.shard_idx), int(args.num_shards)):
                continue
        reviews = row.get('reviews') or []
        # latest-first (descending time)
        reviews = sorted(reviews, key=lambda x: x.get('timestamp') or x.get('date') or 0, reverse=True)
        for r in reviews:
            rv = r.get('rating') if r.get('rating') is not None else (r.get('stars') or r.get('score'))
            try:
                rf = float(rv) if rv is not None else None
            except Exception:
                rf = None
            if rf is None or rf < float(args.min_rating):
                continue
            bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
            summ = r.get('summary') or bid2summary.get(bid) or ''
            ext = r.get('extracted') or (comp_map.get((bid, uid)) if comp_map else {}) or {}
            txt = build_text_review(bid, summ, ext)
            if not txt:
                continue
            rid = make_review_uid(uid, r)
            rows.append((uid, rid, txt))
            user_review_uids.setdefault(uid, []).append(rid)

    if not rows:
        raise ValueError('No like-only interactions found to encode.')

    # Write uid index and rows
    uids = [rid for _, rid, _ in rows]
    with open(os.path.join(args.out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for u in uids:
            f.write(u + '\n')
    with open(os.path.join(args.out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(user_review_uids, f)

    # Debug: show encoder inputs if requested
    if args.debug_print_inputs:
        try:
            k = max(1, int(args.debug_print_k))
            print('[DEBUG] Encoder input sample (first', k, 'of', len(rows), '):')
            for i, (_u, _rid, _txt) in enumerate(rows[:k]):
                print(f'--- sample[{i}] uid={_u} rid={_rid} ---')
                print(_txt)
            dbg_path = os.path.join(args.out_dir, 'debug_inputs_sample.txt')
            with open(dbg_path, 'w', encoding='utf-8') as df:
                for _u, _rid, _txt in rows[:k]:
                    df.write(f'uid={_u} rid={_rid}\n')
                    df.write(_txt)
                    df.write('\n\n')
            print('[DEBUG] Saved encoder input samples to:', dbg_path)
        except Exception:
            pass

    # Encode and write mmap
    dim = enc.config.hidden_size
    mmap_path = os.path.join(args.out_dir, 'interaction_embeds.fp16.mmap')
    mmap = np.memmap(mmap_path, mode='w+', dtype=np.float16, shape=(len(uids), int(dim)))
    bs = int(args.batch_size)
    for i in range(0, len(uids), bs):
        texts = [t for _, _, t in rows[i:i + bs]]
        vecs = encode(texts)
        mmap[i:i + vecs.shape[0], :] = vecs
    mmap.flush()

    meta = {
        'model': args.model,
        'dim': int(dim),
        'normalized': True,
    }
    with open(os.path.join(args.out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    main()

# --- Sharding helpers and merge ---
def _hash_to_shard(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    import hashlib
    h = hashlib.sha1(key.encode('utf-8')).digest()
    val = int.from_bytes(h[:8], byteorder='big', signed=False)
    return val % max(1, num_shards)

def user_belongs_to_shard(user_id: str, shard_idx: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return _hash_to_shard(user_id, num_shards) == int(shard_idx)

def merge_shard_outputs(shard_dirs: List[str], out_dir: str) -> None:
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    shard_dirs = [os.path.abspath(p) for p in shard_dirs]
    shard_dirs = sorted(shard_dirs)
    metas = []
    uids_lists: List[List[str]] = []
    sizes: List[int] = []
    dim = None
    for d in shard_dirs:
        meta_path = os.path.join(d, 'cache_meta.json')
        uids_path = os.path.join(d, 'uids.txt')
        mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        if not (os.path.exists(meta_path) and os.path.exists(uids_path) and os.path.exists(mmap_path)):
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(uids_path, 'r', encoding='utf-8') as f:
            uids = [ln.strip() for ln in f if ln.strip()]
        if dim is None:
            dim = int(meta['dim'])
        metas.append(meta)
        uids_lists.append(uids)
        sizes.append(len(uids))
    total = sum(sizes)
    if dim is None:
        raise ValueError('No shard metas found to merge')
    out_mmap_path = os.path.join(out_dir, 'interaction_embeds.fp16.mmap')
    out_mmap = np.memmap(out_mmap_path, mode='w+', dtype=np.float16, shape=(total, int(dim)))
    offset = 0
    for d, n in zip(shard_dirs, sizes):
        shard_mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        shard = np.memmap(shard_mmap_path, mode='r', dtype=np.float16, shape=(n, int(dim)))
        out_mmap[offset:offset+n, :] = shard[:]
        offset += n
    out_mmap.flush()
    # Merge uids
    with open(os.path.join(out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for uids in uids_lists:
            for u in uids:
                f.write(u + '\n')
    # Merge index.json
    merged_index: Dict[str, List[str]] = {}
    for d in shard_dirs:
        idx_path = os.path.join(d, 'index.json')
        if os.path.exists(idx_path):
            with open(idx_path, 'r', encoding='utf-8') as f:
                part_idx = json.load(f)
            for uid, lst in part_idx.items():
                if not isinstance(lst, list):
                    continue
                merged_index.setdefault(uid, []).extend(lst)
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_index, f)
    # Write meta
    base_meta = metas[0] if metas else {'model': '', 'dim': int(dim), 'normalized': True}
    with open(os.path.join(out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(base_meta, f)

