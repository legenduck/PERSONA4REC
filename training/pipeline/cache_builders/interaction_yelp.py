#!/usr/bin/env python3
import os, json, argparse, gzip, hashlib
import sys, subprocess, tempfile
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


def make_review_uid(user_id: str, r: Dict[str, Any]) -> str:
    bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
    ts = str(r.get('timestamp') or r.get('date') or '')
    # Title not available in Yelp; use business_id + timestamp + stars + first 16 chars of text
    stars = str(r.get('stars') or r.get('rating') or '')
    text = (r.get('text') or '')[:16]
    h = hashlib.sha1(f"{user_id}|{bid}|{ts}|{stars}|{text}".encode('utf-8')).hexdigest()
    return h[:16]


def _hash_to_shard(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    h = hashlib.sha1(key.encode('utf-8')).digest()
    # Use first 8 bytes for stable 64-bit hash
    val = int.from_bytes(h[:8], byteorder='big', signed=False)
    return val % max(1, num_shards)


def user_belongs_to_shard(user_id: str, shard_idx: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return _hash_to_shard(user_id, num_shards) == int(shard_idx)


def merge_shard_outputs(shard_dirs: List[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Normalize and sort shard dirs for deterministic order
    shard_dirs = [os.path.abspath(p) for p in shard_dirs]
    shard_dirs = sorted(shard_dirs)
    metas = []
    uids_lists: List[List[str]] = []
    sizes: List[int] = []
    dim = None
    base_meta = None
    for d in shard_dirs:
        meta_path = os.path.join(d, 'cache_meta.json')
        uids_path = os.path.join(d, 'uids.txt')
        mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        if not (os.path.exists(meta_path) and os.path.exists(uids_path) and os.path.exists(mmap_path)):
            raise FileNotFoundError(f"Missing files in shard dir: {d}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(uids_path, 'r', encoding='utf-8') as f:
            uids = [ln.strip() for ln in f if ln.strip()]
        if dim is None:
            dim = int(meta['dim'])
            base_meta = meta
        else:
            # Validate compatibility
            for k in ['model', 'dim', 'normalized', 'pooling', 'mode']:
                if meta.get(k) != base_meta.get(k):
                    raise ValueError(f"Shard meta mismatch on {k}: {meta.get(k)} != {base_meta.get(k)}")
        metas.append(meta)
        uids_lists.append(uids)
        sizes.append(len(uids))

    total = sum(sizes)
    if dim is None:
        raise ValueError('No shard metas found to merge')

    # Create output mmap
    out_mmap_path = os.path.join(out_dir, 'interaction_embeds.fp16.mmap')
    out_mmap = np.memmap(out_mmap_path, mode='w+', dtype=np.float16, shape=(total, int(dim)))

    # Copy shard mmaps into output
    offset = 0
    for d, n in zip(shard_dirs, sizes):
        shard_mmap_path = os.path.join(d, 'interaction_embeds.fp16.mmap')
        shard = np.memmap(shard_mmap_path, mode='r', dtype=np.float16, shape=(n, int(dim)))
        out_mmap[offset:offset+n, :] = shard[:]
        offset += n
    out_mmap.flush()

    # Merge uids.txt
    with open(os.path.join(out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for uids in uids_lists:
            for u in uids:
                f.write(u + '\n')

    # Merge index.json and rows.jsonl
    merged_index: Dict[str, List[str]] = {}
    with open(os.path.join(out_dir, 'rows.jsonl'), 'w', encoding='utf-8') as rows_out:
        for d in shard_dirs:
            idx_path = os.path.join(d, 'index.json')
            rows_path = os.path.join(d, 'rows.jsonl')
            if os.path.exists(idx_path):
                with open(idx_path, 'r', encoding='utf-8') as f:
                    part_idx = json.load(f)
                for uid, lst in part_idx.items():
                    if not isinstance(lst, list):
                        continue
                    merged_index.setdefault(uid, []).extend(lst)
            if os.path.exists(rows_path):
                with open(rows_path, 'r', encoding='utf-8') as rf:
                    for ln in rf:
                        rows_out.write(ln)
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_index, f)

    # Write meta
    merged_meta = dict(base_meta)
    merged_meta['merged_from'] = shard_dirs
    with open(os.path.join(out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_meta, f)

    print('merge_out_dir:', out_dir)
    print('merged_count:', int(total), 'dim:', int(dim))
    print('files:', ['interaction_embeds.fp16.mmap', 'index.json', 'uids.txt', 'rows.jsonl', 'cache_meta.json'])

def build_text_summary(r: Dict[str, Any], bid2summary: Dict[str, str]) -> str:
    # Summary mode: ONLY item summary (no raw review, no rating)
    bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
    summ = (bid2summary.get(bid) or '').strip()
    parts = ['query: [UserProfile-Interaction]']
    if summ:
        parts.append(f'item_summary: {summ}')
    return '\n'.join(parts)


def build_text_component(r: Dict[str, Any], comp_map: Dict[Tuple[str, str], Dict[str, Any]], bid2summary: Dict[str, str], user_id: str) -> str:
    # Component mode: item summary + components (no raw review, no rating)
    bid = str(r.get('business_id') or r.get('asin') or r.get('item_id') or r.get('parent_asin') or '')
    ext = comp_map.get((bid, user_id)) or {}
    summ = (bid2summary.get(bid) or '').strip()
    parts = ['query: [UserProfile-Interaction]']
    if summ:
        parts.append(f'item_summary: {summ}')
    if isinstance(ext, dict):
        if ext.get('cuisine_category'):
            parts.append(f"Cuisine Category: {ext.get('cuisine_category')}")
        if ext.get('cuisine_category_reason'):
            parts.append(f"Cuisine Category Reason: {ext.get('cuisine_category_reason')}")
        if ext.get('specific_dishes_or_attributes'):
            parts.append(f"Specific Dishes/Attributes: {ext.get('specific_dishes_or_attributes')}")
        if ext.get('visit_purpose'):
            parts.append(f"Visit Purpose: {ext.get('visit_purpose')}")
        if ext.get('visit_purpose_reason'):
            parts.append(f"Visit Purpose Reason: {ext.get('visit_purpose_reason')}")
        if ext.get('quality_criteria'):
            parts.append(f"Valued Quality Criteria: {ext.get('quality_criteria')}")
        if ext.get('visit_context'):
            parts.append(f"Visit Context: {ext.get('visit_context')}")
        if ext.get('visit_context_reason'):
            parts.append(f"Visit Context Reason: {ext.get('visit_context_reason')}")
    return '\n'.join(parts)


def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def orchestrate_multi_gpu(args) -> None:
    # Pre-scan history to collect ordered user_ids
    user_ids: List[str] = []
    seen = set()
    for row in read_jsonl_auto(args.history):
        uid = str(row.get('user_id') or '')
        if uid and uid not in seen:
            seen.add(uid)
            user_ids.append(uid)
    if not user_ids:
        raise ValueError('No users found in history for sharding')

    num = int(args.num_gpus)
    # Determine GPU ids to use
    available = []
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            available = list(range(count))
        except Exception:
            available = []
    if available and len(available) < num:
        # Still proceed; will recycle ids
        pass

    # Split user_ids into contiguous chunks
    chunks_list: List[List[str]] = []
    n = len(user_ids)
    for i in range(num):
        start = (n * i) // num
        end = (n * (i + 1)) // num
        chunks_list.append(user_ids[start:end])

    # Prepare per-shard out dirs and allowlists
    shard_dirs: List[str] = []
    allow_files: List[str] = []
    for i in range(num):
        d = os.path.join(args.out_dir, f'shard_{i}')
        os.makedirs(d, exist_ok=True)
        shard_dirs.append(d)
        allow_path = os.path.join(d, 'allow_users.txt')
        with open(allow_path, 'w', encoding='utf-8') as f:
            for u in chunks_list[i]:
                f.write(u + '\n')
        allow_files.append(allow_path)

    # Spawn subprocesses
    procs = []
    this_script = os.path.abspath(sys.argv[0])
    for i in range(num):
        gpu = available[i % max(1, len(available))] if available else ''
        cmd = [sys.executable, this_script,
               '--history', args.history,
               '--aspects', args.aspects,
               '--mode', args.mode,
               '--model', args.model,
               '--out_dir', shard_dirs[i],
               '--batch_size', str(args.batch_size),
               '--max_len', str(args.max_len),
               '--checkpoint_dir', args.checkpoint_dir,
               '--proj_checkpoint', args.proj_checkpoint,
               '--item_summary', args.item_summary,
               '--shard_idx', str(i), '--num_shards', str(num),
               '--internal_shard_run', '--user_allowlist', allow_files[i]
        ]
        if args.strict_components:
            cmd.append('--strict_components')
        if gpu != '':
            cmd.extend(['--gpu_id', str(gpu)])
        env = os.environ.copy()
        # Do not override CUDA_VISIBLE_DEVICES unless necessary; rely on --gpu_id
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for all
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise RuntimeError(f'Sharded processes failed with codes: {codes}')

    # Merge
    merge_shard_outputs(shard_dirs, args.out_dir)

    # Optionally: keep shard dirs for inspection; comment out to remove
    # for d in shard_dirs:
    #     shutil.rmtree(d, ignore_errors=True)

def main_with_args(**kwargs):
    """
    Python API: Build Yelp interaction cache with keyword arguments.
    
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
    args.aspects = kwargs.get('aspects', '')
    args.mode = kwargs.get('mode', 'summary')
    args.model = kwargs.get('model', 'BAAI/bge-m3')
    args.out_dir = kwargs.get('out_dir', '')
    args.batch_size = int(kwargs.get('batch_size', 64))
    args.max_len = int(kwargs.get('max_len', 512))
    args.checkpoint_dir = kwargs.get('checkpoint_dir', '')
    args.proj_checkpoint = kwargs.get('proj_checkpoint', '')
    args.item_summary = kwargs.get('item_summary', '')
    args.strict_aspects = kwargs.get('strict_aspects', False)
    args.debug_print_inputs = kwargs.get('debug_print_inputs', False)
    args.debug_print_k = int(kwargs.get('debug_print_k', 5))
    args.num_shards = int(kwargs.get('num_shards', 1))
    args.shard_idx = int(kwargs.get('shard_idx', 0))
    args.gpu_id = kwargs.get('gpu_id', '')
    args.merge_from = kwargs.get('merge_from', None)
    args.num_gpus = int(kwargs.get('num_gpus', 1))
    args.internal_shard_run = kwargs.get('internal_shard_run', False)
    args.user_allowlist = kwargs.get('user_allowlist', '')
    
    # Run main logic (insert existing main() code here)
    _run_main_logic(args)


def main():
    ap = argparse.ArgumentParser(description='Build Yelp per-interaction embeddings cache (like-only)')
    ap.add_argument('--history', required=True, help='user_review_history_full.json.gz (JSON per user with reviews list)')
    ap.add_argument('--aspects', default='', help='Aspects file (2.5_grouped.jsonl)')
    ap.add_argument('--mode', default='summary', choices=['summary','aspects'])
    ap.add_argument('--model', default='BAAI/bge-m3')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--checkpoint_dir', default='')
    ap.add_argument('--proj_checkpoint', default='')
    ap.add_argument('--item_summary', default='', help='item summaries jsonl (business_id,item_summary)')
    ap.add_argument('--strict_aspects', action='store_true', help='If set, invalid aspect entries raise error. Otherwise they are skipped with a count.')
    # Debug: print example LLM inputs
    ap.add_argument('--debug_print_inputs', action='store_true', help='Print sample input texts that will be encoded')
    ap.add_argument('--debug_print_k', type=int, default=5, help='How many sample inputs to print')
    # Multi-GPU / sharding
    ap.add_argument('--num_shards', type=int, default=1, help='Total number of shards (e.g., number of GPUs)')
    ap.add_argument('--shard_idx', type=int, default=0, help='Shard index for this process [0..num_shards-1]')
    ap.add_argument('--gpu_id', default='', help='Specific GPU id to use, e.g., 0 or 1. Empty = auto')
    # Merge mode
    ap.add_argument('--merge_from', nargs='*', default=None, help='List of shard cache dirs to merge into --out_dir')
    # Orchestration
    ap.add_argument('--num_gpus', type=int, default=1, help='If >1, auto-shard users into contiguous blocks and run in parallel')
    ap.add_argument('--internal_shard_run', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--user_allowlist', default='', help=argparse.SUPPRESS)
    args = ap.parse_args()
    
    _run_main_logic(args)


def _run_main_logic(args):
    """Internal function containing the main logic."""
    os.makedirs(args.out_dir, exist_ok=True)

    # Merge-only mode
    if args.merge_from:
        return merge_shard_outputs(args.merge_from, args.out_dir)

    # Auto multi-GPU orchestration (preserve single-GPU order)
    if int(args.num_gpus) > 1 and not bool(args.internal_shard_run):
        return orchestrate_multi_gpu(args)

    # Load aspects map for Yelp (Schema A ONLY): keyed by (business_id, user_id)
    # Schema A: { business_id, reviews: [{ user_id, extracted: {...} }] }
    aspects_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if args.mode == 'aspects' and args.aspects:
        line_idx = 0
        invalid_count = 0
        for row in read_jsonl_auto(args.aspects):
            line_idx += 1
            bid_val = row.get('business_id')
            if not bid_val:
                raise ValueError(f"aspects line {line_idx}: missing business_id")
            bid = str(bid_val)
            reviews = row.get('reviews')
            if not isinstance(reviews, list):
                raise ValueError(f"aspects line {line_idx}: expected Schema A with 'reviews' list")
            for rev in reviews:
                if not isinstance(rev, dict):
                    if args.strict_aspects:
                        raise ValueError(f"aspects line {line_idx}: review entry must be object")
                    invalid_count += 1
                    continue
                uid_val = rev.get('user_id')
                if not uid_val:
                    if args.strict_aspects:
                        raise ValueError(f"aspects line {line_idx}: review missing user_id")
                    invalid_count += 1
                    continue
                uid = str(uid_val)
                ext = rev.get('extracted')
                if not isinstance(ext, dict):
                    if args.strict_aspects:
                        raise ValueError(f"aspects line {line_idx}: review.extracted must be object")
                    invalid_count += 1
                    continue
                aspects_map[(bid, uid)] = ext
        if invalid_count:
            print(f"[aspects] skipped invalid entries: {invalid_count}")

    # Build item_id -> summary map (robust to different id fields)
    bid2summary: Dict[str, str] = {}
    for row in read_jsonl_auto(args.item_summary):
        bid = row.get('business_id') or row.get('asin') or row.get('item_id') or row.get('parent_asin')
        if not bid:
            continue
        bid2summary[str(bid)] = row.get('item_summary') or row.get('Item_summary') or ''

    # Prepare encoder
    # Select device (respect --gpu_id if provided)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and str(args.gpu_id).strip() != '':
        try:
            torch.cuda.set_device(int(str(args.gpu_id)))
            device = f"cuda:{int(str(args.gpu_id))}"
        except Exception:
            device = 'cuda'
    use_st = os.path.exists(os.path.join(args.model, 'modules.json'))
    st_model = None
    tok = None
    enc = None
    if use_st:
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(args.model)
            try:
                if isinstance(args.max_len, int) and args.max_len > 0:
                    st_model.max_seq_length = args.max_len
            except Exception:
                pass
            st_model = st_model.to(device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer from {args.model}: {e}")
    else:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        enc = AutoModel.from_pretrained(args.model, use_safetensors=True, trust_remote_code=True).to(device).eval()

    # Optional projection head
    if use_st and st_model is not None:
        try:
            emb_dim = int(st_model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: typical BGE hidden size
            emb_dim = 1024
    else:
        emb_dim = enc.config.hidden_size
    proj_layer = torch.nn.Identity().to(device).eval()
    out_dim = emb_dim
    checkpoint_path = ''
    if args.proj_checkpoint:
        checkpoint_path = args.proj_checkpoint
    elif args.checkpoint_dir:
        candidate = os.path.join(args.checkpoint_dir, 'dual_extra.pt')
        if os.path.exists(candidate):
            checkpoint_path = candidate
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            extra = torch.load(checkpoint_path, map_location='cpu')
            state = extra.get('proj_user') or extra.get('proj_persona')
            if state is not None and isinstance(state, dict) and 'weight' in state:
                w = state['weight']
                proj_dim = int(w.shape[0])
                layer = torch.nn.Linear(emb_dim, proj_dim, bias=False)
                layer.load_state_dict(state)
                proj_layer = layer.to(device).eval()
                out_dim = proj_dim
        except Exception:
            pass

    # Iterate and collect interactions (like-only), also build user->ordered review_uids
    user_review_uids: Dict[str, List[str]] = {}
    rows: List[Tuple[str, str, str]] = []  # (user_id, review_uid, text)

    allow_set = None
    if args.user_allowlist:
        with open(args.user_allowlist, 'r', encoding='utf-8') as f:
            allow_set = set(ln.strip() for ln in f if ln.strip())
    for row in read_jsonl_auto(args.history):
        uid = str(row.get('user_id') or '')
        # Filter by allowlist if provided; else use shard hash filter when num_shards>1
        if allow_set is not None:
            if uid not in allow_set:
                continue
        elif int(args.num_shards) > 1:
            if not user_belongs_to_shard(uid, args.shard_idx, args.num_shards):
                continue
        reviews = row.get('reviews') or []
        # sort by timestamp/date desc
        reviews = sorted(reviews, key=lambda x: x.get('timestamp') or x.get('date') or 0, reverse=True)
        for r in reviews:
            stars = r.get('stars') if r.get('stars') is not None else r.get('rating')
            try:
                rv = float(stars) if stars is not None else None
            except Exception:
                rv = None
            if rv is None or rv < 3:
                continue
            review_uid = make_review_uid(uid, r)
            if args.mode == 'aspects':
                text = build_text_component(r, aspects_map, bid2summary, uid)
            else:
                text = build_text_summary(r, bid2summary)
            rows.append((uid, review_uid, text))
            user_review_uids.setdefault(uid, []).append(review_uid)

    if not rows:
        raise ValueError('No like-only interactions found to encode.')

    # Encode in batches
    def encode(texts: List[str]) -> np.ndarray:
        if use_st and st_model is not None:
            vecs = st_model.encode(
                texts,
                batch_size=max(1, int(args.batch_size)),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            emb = torch.from_numpy(vecs).to(device=device, dtype=torch.float32)
            if not isinstance(proj_layer, torch.nn.Identity):
                emb = proj_layer(emb)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-6)
            return emb.detach().cpu().to(torch.float16).numpy()
        else:
            batch = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors='pt')
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
                if not isinstance(proj_layer, torch.nn.Identity):
                    emb = proj_layer(emb)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-6)
            return emb.detach().cpu().to(torch.float16).numpy()

    uids = [u for _, u, _ in rows]
    texts = [t for _, _, t in rows]

    # Debug: show what goes into the encoder (LLM input texts)
    if args.debug_print_inputs:
        k = max(1, int(args.debug_print_k))
        print('[DEBUG] Encoder input sample (first', k, 'of', len(texts), '):')
        for i, s in enumerate(texts[:k]):
            try:
                print(f'--- sample[{i}] ---')
                print(s)
            except Exception:
                pass
        try:
            dbg_path = os.path.join(args.out_dir, 'debug_inputs_sample.txt')
            with open(dbg_path, 'w', encoding='utf-8') as df:
                for s in texts[:k]:
                    df.write(s)
                    df.write('\n\n')
            print('[DEBUG] Saved encoder input samples to:', dbg_path)
        except Exception:
            pass
    mmap_path = os.path.join(args.out_dir, 'interaction_embeds.fp16.mmap')
    mmap = np.memmap(mmap_path, mode='w+', dtype=np.float16, shape=(len(texts), out_dim))
    for chunk in chunks(list(range(len(texts))), args.batch_size):
        vecs = encode([texts[i] for i in chunk])
        mmap[chunk, :] = vecs
    mmap.flush()

    # Save indices
    with open(os.path.join(args.out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump({uid: lst for uid, lst in user_review_uids.items()}, f)
    with open(os.path.join(args.out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for u in uids:
            f.write(u + '\n')
    with open(os.path.join(args.out_dir, 'rows.jsonl'), 'w', encoding='utf-8') as f:
        for (uid, rid, _text) in rows:
            f.write(json.dumps({'user_id': uid, 'review_uid': rid}) + '\n')
    with open(os.path.join(args.out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.model,
            'dim': int(out_dim),
            'normalized': True,
            'pooling': 'attn_mean',
            'mode': args.mode,
            'uses_projection': not isinstance(proj_layer, torch.nn.Identity)
        }, f)

    print('cache_dir:', args.out_dir)
    print('count:', len(texts), 'dim:', int(out_dim))
    print('files:', ['interaction_embeds.fp16.mmap', 'index.json', 'uids.txt', 'rows.jsonl', 'cache_meta.json'])


def orchestrate_multi_gpu(args) -> None:
    # Pre-scan history to collect ordered user_ids
    user_ids: List[str] = []
    seen = set()
    for row in read_jsonl_auto(args.history):
        uid = str(row.get('user_id') or '')
        if uid and uid not in seen:
            seen.add(uid)
            user_ids.append(uid)
    if not user_ids:
        raise ValueError('No users found in history for sharding')

    num = int(args.num_gpus)
    # Determine GPU ids to use
    available = []
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            available = list(range(count))
        except Exception:
            available = []
    if available and len(available) < num:
        # Still proceed; will recycle ids
        pass

    # Split user_ids into contiguous chunks
    chunks_list: List[List[str]] = []
    n = len(user_ids)
    for i in range(num):
        start = (n * i) // num
        end = (n * (i + 1)) // num
        chunks_list.append(user_ids[start:end])

    # Prepare per-shard out dirs and allowlists
    shard_dirs: List[str] = []
    allow_files: List[str] = []
    for i in range(num):
        d = os.path.join(args.out_dir, f'shard_{i}')
        os.makedirs(d, exist_ok=True)
        shard_dirs.append(d)
        allow_path = os.path.join(d, 'allow_users.txt')
        with open(allow_path, 'w', encoding='utf-8') as f:
            for u in chunks_list[i]:
                f.write(u + '\n')
        allow_files.append(allow_path)

    # Spawn subprocesses
    procs = []
    this_script = os.path.abspath(sys.argv[0])
    for i in range(num):
        gpu = available[i % max(1, len(available))] if available else ''
        cmd = [sys.executable, this_script,
               '--history', args.history,
               '--aspects', args.aspects,
               '--mode', args.mode,
               '--model', args.model,
               '--out_dir', shard_dirs[i],
               '--batch_size', str(args.batch_size),
               '--max_len', str(args.max_len),
               '--checkpoint_dir', args.checkpoint_dir,
               '--proj_checkpoint', args.proj_checkpoint,
               '--item_summary', args.item_summary,
               '--shard_idx', str(i), '--num_shards', str(num),
               '--internal_shard_run', '--user_allowlist', allow_files[i]
        ]
        if args.strict_components:
            cmd.append('--strict_components')
        if gpu != '':
            cmd.extend(['--gpu_id', str(gpu)])
        env = os.environ.copy()
        # Do not override CUDA_VISIBLE_DEVICES unless necessary; rely on --gpu_id
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for all
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise RuntimeError(f'Sharded processes failed with codes: {codes}')

    # Merge
    merge_shard_outputs(shard_dirs, args.out_dir)

    # Optionally: keep shard dirs for inspection; comment out to remove
    # for d in shard_dirs:
    #     shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    main()
