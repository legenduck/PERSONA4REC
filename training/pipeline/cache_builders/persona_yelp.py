import os, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# No default paths - all paths must be provided by user


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_text(p: dict, fields: list) -> str:
    parts = ['passage: [Persona]']
    for key in fields:
        if key == 'Name':
            parts.append(f"Name: {(p.get('Name') or '').strip()}")
        elif key == 'Description':
            parts.append(f"Description: {(p.get('Description') or '').strip()}")
        elif key == 'item_summary':
            # Prefer Item_Summary; fall back to common variants
            _summ = (
                p.get('Item_Summary')
                or p.get('Item_summary')
                or p.get('item_summary')
                or p.get('item_description')
                or ''
            )
            parts.append(f"item_summary: {_summ.strip()}")
        elif key == 'Preference_Rationale':
            parts.append(f"Preference Rationale: {(p.get('Preference_Rationale') or '').strip()}")
        else:
            continue
    return '\n'.join(parts)


def main_with_args(**kwargs):
    """
    Python API: Build Yelp persona cache with keyword arguments.
    
    Args:
        model (str): Model path
        out_dir (str): Output directory
        persona_path (str): Path to persona file
        fields (str): Comma-separated fields (N,D,I,R)
        batch_size (int): Batch size
        persona_max_len (int): Max sequence length
        **kwargs: Additional arguments
    """
    # Create a namespace object from kwargs
    class Args:
        pass
    args = Args()
    
    # Set attributes from kwargs
    args.model = kwargs.get('model', '')
    args.out_dir = kwargs.get('out_dir', '')
    args.fields = kwargs.get('fields', 'N,D,I')
    args.batch_size = int(kwargs.get('batch_size', 64))
    args.persona_max_len = int(kwargs.get('persona_max_len', 512))
    args.checkpoint_dir = kwargs.get('checkpoint_dir', '')
    args.proj_checkpoint = kwargs.get('proj_checkpoint', '')
    args.persona_path = kwargs.get('persona_path', '')
    
    # Run main logic
    _run_main_logic(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fields', required=True, help="Comma-separated subset: Name,Description,item_summary,Preference_Rationale or N,D,I,R")
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--persona_max_len', type=int, default=512)
    ap.add_argument('--checkpoint_dir', default='')
    ap.add_argument('--proj_checkpoint', default='')
    ap.add_argument('--persona_path', required=True, help='Path to persona jsonl (expects Item_Summary; falls back to item_summary/Item_summary/item_description)')
    args = ap.parse_args()
    
    _run_main_logic(args)


def _run_main_logic(args):
    """Internal function containing the main logic."""
    fields_in = [x.strip() for x in args.fields.split(',') if x.strip()]
    alias = {'N': 'Name', 'D': 'Description', 'I': 'item_summary', 'R': 'Preference_Rationale'}
    valid = {'Name', 'Description', 'item_summary', 'Preference_Rationale'}
    fields = []
    for f in fields_in:
        if f in valid:
            fields.append(f)
            continue
        up = f.upper()
        if up in alias:
            fields.append(alias[up])
            continue
        raise ValueError(f"Invalid field: {f}. Allowed: {sorted(list(valid))} or abbreviations N,D,I,R")
    if not fields:
        raise ValueError('At least one field must be specified via --fields')

    os.makedirs(args.out_dir, exist_ok=True)

    # Load personas
    id2text = {}
    bid2uids = {}
    persona_src = args.persona_path
    for row in read_jsonl(persona_src):
        business_id = row.get('business_id')
        if not business_id:
            continue
        per_list = row.get('personas')
        if not isinstance(per_list, list) or not per_list:
            raise ValueError('persona jsonl row must contain non-empty "personas" array')
        for p in per_list:
            uid = p.get('unique_persona_id')
            if not uid:
                raise ValueError('unique_persona_id is required for all personas')
            txt = build_text(p, fields)
            id2text[uid] = txt
            bid2uids.setdefault(business_id, []).append(uid)

    # Save business_id->uids index
    with open(os.path.join(args.out_dir, 'business_id_to_uids.json'), 'w', encoding='utf-8') as f:
        json.dump(bid2uids, f)

    # Prepare model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_st = os.path.exists(os.path.join(args.model, 'modules.json'))
    st_model = None
    tok = None
    enc = None
    if use_st:
        # SentenceTransformers checkpoint: use SentenceTransformer.encode for consistency
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(args.model)
            try:
                if isinstance(args.persona_max_len, int) and args.persona_max_len > 0:
                    st_model.max_seq_length = args.persona_max_len
            except Exception:
                pass
            st_model = st_model.to(device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer from {args.model}: {e}")
    else:
        # HuggingFace model: use AutoModel + attention-weighted mean pooling
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        enc = AutoModel.from_pretrained(args.model, use_safetensors=True, trust_remote_code=True).to(device)
        enc.eval()

    uids = list(id2text.keys())
    if use_st and st_model is not None:
        try:
            dim = int(st_model.get_sentence_embedding_dimension())
        except Exception:
            dim = 1024
    else:
        try:
            dim = enc.module.config.hidden_size if isinstance(enc, torch.nn.DataParallel) else enc.config.hidden_size
        except Exception:
            # Fallback in case model doesn't expose config
            dim = getattr(getattr(enc, 'module', enc), 'config', {}).get('hidden_size', 1024)

    # Optional projection head
    proj_layer = torch.nn.Identity().to(device)
    proj_dim = 0
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
                layer = torch.nn.Linear(dim, proj_dim, bias=False)
                layer.load_state_dict(state)
                proj_layer = layer.to(device).eval()
        except Exception:
            pass
    out_dim = proj_dim if proj_dim > 0 else dim

    dtype = np.float16
    if len(uids) == 0:
        raise ValueError('No personas found to encode (uids list is empty). Check input file and fields.')
    mmap_path = os.path.join(args.out_dir, 'persona_embeds.fp16.mmap')
    mmap = np.memmap(mmap_path, mode='w+', dtype=dtype, shape=(len(uids), out_dim))

    def encode(texts):
        if use_st and st_model is not None:
            # The ST pipeline already contains Pooling/Normalize modules. Avoid double-normalize.
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
            batch = tok(texts, padding=True, truncation=True, max_length=args.persona_max_len, return_tensors='pt')
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

    uid_to_index = {u: i for i, u in enumerate(uids)}
    for chunk_uids in chunks(uids, args.batch_size):
        texts = [id2text[u] for u in chunk_uids]
        vecs = encode(texts)
        idxs = [uid_to_index[u] for u in chunk_uids]
        mmap[idxs, :] = vecs
    mmap.flush()

    # Save cache meta
    meta = {
        'model': args.model,
        'dim': out_dim,
        'uses_projection': bool(proj_dim and proj_dim > 0),
        'checkpoint': checkpoint_path,
        'pooling': 'attn_mean',
        'normalized': True,
        'includes_item_summary': ('item_summary' in fields),
        'ablation_fields': fields,
        'text_template': '\n'.join(['passage: [Persona]'] + [
            'Name: {Name}' if f=='Name' else (
            'Description: {Description}' if f=='Description' else (
            'item_summary: {Item_Summary}' if f=='item_summary' else (
            'Preference Rationale: {Preference_Rationale}' if f=='Preference_Rationale' else ''
        ))) for f in fields if f in {'Name','Description','item_summary','Preference_Rationale'}])
    }
    with open(os.path.join(args.out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    # Save uid order
    with open(os.path.join(args.out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for u in uids:
            f.write(u + '\n')

    # Ensure index file exists (written above)
    with open(os.path.join(args.out_dir, 'business_id_to_uids.json'), 'r', encoding='utf-8') as f:
        pass

    print('cache_dir:', args.out_dir)
    print('count:', len(uids), 'dim:', out_dim)
    print('fields:', fields)
    print('files:', ['business_id_to_uids.json', 'uids.txt', 'persona_embeds.fp16.mmap', 'cache_meta.json'])

    # Integrity checks: ensure all unique_persona_id are cached and indexed
    try:
        # Reload written files
        with open(os.path.join(args.out_dir, 'uids.txt'), 'r', encoding='utf-8') as f:
            uids_written = [ln.strip() for ln in f if ln.strip()]
        with open(os.path.join(args.out_dir, 'business_id_to_uids.json'), 'r', encoding='utf-8') as f:
            bid2uids_written = json.load(f)

        # Expected set from input personas
        expected_uids = set(uids)
        written_set = set(uids_written)
        mapped_set = set([uid for lst in bid2uids_written.values() for uid in lst])

        errors = []
        if len(uids_written) != len(uids):
            errors.append(f"uids.txt length mismatch: {len(uids_written)} vs expected {len(uids)}")
        missing_in_uids = sorted(list(expected_uids - written_set))
        if missing_in_uids:
            errors.append(f"missing in uids.txt: {len(missing_in_uids)} (e.g., {missing_in_uids[:3]})")
        missing_in_map = sorted(list(expected_uids - mapped_set))
        if missing_in_map:
            errors.append(f"missing in business_id_to_uids.json: {len(missing_in_map)} (e.g., {missing_in_map[:3]})")

        # Memmap rows count check
        try:
            _m = np.memmap(os.path.join(args.out_dir, 'persona_embeds.fp16.mmap'), mode='r', dtype=np.float16)
            rows = int(_m.size // out_dim) if out_dim > 0 else 0
            if rows != len(uids):
                errors.append(f"memmap rows mismatch: {rows} vs expected {len(uids)}")
            del _m
        except Exception as e:
            errors.append(f"memmap open failed: {e}")

        if errors:
            raise RuntimeError("[VERIFY] Persona cache integrity check failed: " + "; ".join(errors))
        else:
            print('[VERIFY] Persona cache integrity OK:', {
                'total_personas': len(uids),
                'unique_uids_txt': len(written_set),
                'unique_bmap_uids': len(mapped_set),
                'dim': out_dim
            })
    except Exception as e:
        print(f"[VERIFY] Warning: verification encountered an issue: {e}")


if __name__ == '__main__':
    main()



