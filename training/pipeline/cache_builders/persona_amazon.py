#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Any
import numpy as np


def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def build_text(p: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = ['passage: [Persona]']
    name = p.get('Name') or p.get('name')
    desc = p.get('Description') or p.get('description')
    item_sum = p.get('Item_Summary') or p.get('Item_summary') or p.get('item_summary')
    pref_rat = p.get('Preference_Rationale') or p.get('preference_rationale')
    for f in fields:
        if f == 'Name' and isinstance(name, str) and name.strip():
            parts.append(f"Name: {name.strip()}")
        elif f == 'Description' and isinstance(desc, str) and desc.strip():
            parts.append(f"Description: {desc.strip()}")
        elif f == 'item_summary' and isinstance(item_sum, str) and item_sum.strip():
            parts.append(f"Item Summary: {item_sum.strip()}")
        elif f == 'Preference_Rationale' and isinstance(pref_rat, str) and pref_rat.strip():
            parts.append(f"Preference Rationale: {pref_rat.strip()}")
    return '\n'.join(parts)


def encode_texts_st(model_dir: str, texts: List[str], max_len: int, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    st = SentenceTransformer(model_dir)
    if isinstance(max_len, int) and max_len > 0:
        try:
            st.max_seq_length = max_len
        except Exception:
            pass
    vecs = st.encode(texts, batch_size=max(1, int(batch_size)), convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    emb = torch.from_numpy(vecs).to(dtype=torch.float32)
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-6)
    return emb.detach().cpu().to(torch.float16).numpy()


def main_with_args(**kwargs):
    """
    Python API: Build Amazon persona cache with keyword arguments.
    
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
    args.persona_path = kwargs.get('persona_path', '')
    args.fields = kwargs.get('fields', 'N,D,I')
    args.batch_size = int(kwargs.get('batch_size', 1024))
    args.persona_max_len = int(kwargs.get('persona_max_len', 512))
    
    # Run main logic
    _run_main_logic(args)


def main():
    ap = argparse.ArgumentParser(description='Build Amazon persona cache (standalone)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--persona_path', required=True, help='persona_catalog_uid.jsonl')
    ap.add_argument('--fields', default='N,D,I', help='subset: N,D,I,R => Name,Description,item_summary,Preference_Rationale')
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--persona_max_len', type=int, default=512)
    args = ap.parse_args()
    
    _run_main_logic(args)


def _run_main_logic(args):
    """Internal function containing the main logic."""
    alias = {'N': 'Name', 'D': 'Description', 'I': 'item_summary', 'R': 'Preference_Rationale'}
    fields_in = [x.strip() for x in str(args.fields).split(',') if x.strip()]
    fields: List[str] = []
    for f in fields_in:
        if f in ('Name', 'Description', 'item_summary', 'Preference_Rationale'):
            fields.append(f)
        elif f.upper() in alias:
            fields.append(alias[f.upper()])
    if not fields:
        fields = ['Name', 'Description', 'item_summary']

    os.makedirs(args.out_dir, exist_ok=True)

    # Load personas
    id2text: Dict[str, str] = {}
    bid2uids: Dict[str, List[str]] = {}
    for row in read_jsonl(args.persona_path):
        bid = row.get('business_id') or row.get('asin') or row.get('item_id') or row.get('parent_asin')
        personas = row.get('personas') or []
        if not isinstance(bid, str) or not isinstance(personas, list):
            continue
        for p in personas:
            uid = p.get('unique_persona_id') or p.get('unique_id') or p.get('ID')
            if not isinstance(uid, str):
                continue
            txt = build_text(p, fields)
            if not txt:
                continue
            id2text[uid] = txt
            bid2uids.setdefault(bid, []).append(uid)

    # Save mapping
    with open(os.path.join(args.out_dir, 'business_id_to_uids.json'), 'w', encoding='utf-8') as f:
        json.dump(bid2uids, f)

    if not id2text:
        raise RuntimeError('No personas to encode from persona_path')
    uids = list(id2text.keys())

    # Determine embedding dim via a probe
    try:
        from sentence_transformers import SentenceTransformer
        dim = int(SentenceTransformer(args.model).get_sentence_embedding_dimension())
    except Exception:
        dim = 1024

    mmap_path = os.path.join(args.out_dir, 'persona_embeds.fp16.mmap')
    mmap = np.memmap(mmap_path, mode='w+', dtype=np.float16, shape=(len(uids), int(dim)))

    # Encode in batches
    batch = int(args.batch_size)
    order_texts: List[str] = []
    for i in range(0, len(uids), batch):
        chunk_uids = uids[i:i + batch]
        texts = [id2text[u] for u in chunk_uids]
        vecs = encode_texts_st(args.model, texts, int(args.persona_max_len), batch)
        mmap[i:i + vecs.shape[0], :] = vecs
        order_texts.extend(texts)
    mmap.flush()

    # Save meta and uids
    meta = {
        'model': args.model,
        'dim': int(dim),
        'normalized': True,
        'ablation_fields': fields,
    }
    with open(os.path.join(args.out_dir, 'cache_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    with open(os.path.join(args.out_dir, 'uids.txt'), 'w', encoding='utf-8') as f:
        for u in uids:
            f.write(u + '\n')

    # Basic verify
    try:
        with open(os.path.join(args.out_dir, 'uids.txt'), 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) != len(uids):
            print('[VERIFY] Warning: uids.txt count mismatch', file=sys.stderr)
    except Exception:
        pass


if __name__ == '__main__':
    main()

