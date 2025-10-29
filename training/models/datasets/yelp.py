import json
import gzip
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class ProfileTrainingTuple:
    user_id: str
    persona_text: str
    recent_item_texts: List[str]
    recent_item_weights: List[float]


def build_persona_text(
    name: Optional[str],
    description: Optional[str],
    item_summary: Optional[str],
    preference_rationale: Optional[str],
) -> str:
    parts: List[str] = ['passage: [Persona]']
    if isinstance(name, str) and name.strip():
        parts.append(f"Name: {name.strip()}")
    if isinstance(description, str) and description.strip():
        parts.append(f"Description: {description.strip()}")
    if isinstance(item_summary, str) and item_summary.strip():
        parts.append(f"Item Summary: {item_summary.strip()}")
    if isinstance(preference_rationale, str) and preference_rationale.strip():
        parts.append(f"Preference Rationale: {preference_rationale.strip()}")
    return '\n'.join(parts)


def build_interaction_text(
    item_summary: Optional[str],
    components: Optional[Dict[str, Any]],
    title: Optional[str] = None,
    business_id: Optional[str] = None,
) -> str:
    parts: List[str] = ['query: [UserProfile-Interaction]']
    if isinstance(business_id, str) and business_id.strip():
        parts.append(f"Business ID: {business_id.strip()}")
    if isinstance(components, dict):
        if components.get('cuisine_category'):
            parts.append(f"Cuisine Category: {components.get('cuisine_category')}")
        if components.get('cuisine_category_reason'):
            parts.append(f"Cuisine Category Reason: {components.get('cuisine_category_reason')}")
        if components.get('specific_dishes_or_attributes'):
            parts.append(f"Specific Dishes: {components.get('specific_dishes_or_attributes')}")
        if components.get('visit_purpose'):
            parts.append(f"Visit Purpose: {components.get('visit_purpose')}")
        if components.get('visit_purpose_reason'):
            parts.append(f"Visit Purpose Reason: {components.get('visit_purpose_reason')}")
        if components.get('quality_criteria'):
            parts.append(f"Quality Criteria: {components.get('quality_criteria')}")
        if components.get('visit_context'):
            parts.append(f"Visit Context: {components.get('visit_context')}")
        if components.get('visit_context_reason'):
            parts.append(f"Visit Context Reason: {components.get('visit_context_reason')}")
    return '\n'.join(parts)


class YelpPersonaDataset:
    def __init__(
        self,
        selected_persona_path: str,
        persona_catalog_path: Optional[str],
        loo_history_path: Optional[str],
        *,
        k_recent: int = 6,
        gamma: float = 0.45,
        min_rating: float = 3.0,
        debug: bool = False,
    ) -> None:
        self.k_recent = int(k_recent)
        self.gamma = float(gamma)
        self.min_rating = float(min_rating)

        # persona catalog
        self.persona_catalog: Dict[str, Dict[str, Optional[str]]] = {}
        if persona_catalog_path:
            open_pc = gzip.open if str(persona_catalog_path).endswith('.gz') else open
            with open_pc(persona_catalog_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        o = json.loads(s)
                    except Exception:
                        continue
                    personas = o.get('personas') or []
                    if not isinstance(personas, list):
                        continue
                    for p in personas:
                        upid = p.get('unique_persona_id')
                        if not isinstance(upid, str):
                            continue
                        self.persona_catalog[upid] = {
                            'Name': p.get('Name'),
                            'Description': p.get('Description'),
                            'Item_Summary': p.get('Item_Summary'),
                            'Preference_Rationale': p.get('Preference_Rationale'),
                        }

        # history
        self.user2reviews: Dict[str, List[Dict[str, Any]]] = {}
        if loo_history_path:
            open_lh = gzip.open if str(loo_history_path).endswith('.gz') else open
            with open_lh(loo_history_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        o = json.loads(s)
                    except Exception:
                        continue
                    uid = o.get('user_id')
                    if not isinstance(uid, str):
                        continue
                    reviews = o.get('reviews') or []
                    items: List[Dict[str, Any]] = []
                    for r in reviews:
                        bid = r.get('item_id')
                        if not isinstance(bid, str):
                            continue
                        title = r.get('review_title')
                        item_sum = r.get('summary')
                        comps_raw = r.get('extracted')
                        comps: Dict[str, Any] = {}
                        if isinstance(comps_raw, dict):
                            comps = comps_raw
                        elif isinstance(comps_raw, str):
                            try:
                                comps = json.loads(comps_raw)
                            except Exception:
                                comps = {}
                        rating = r.get('rating')
                        try:
                            rating_f = float(rating) if rating is not None else None
                        except Exception:
                            rating_f = None
                        ts = r.get('timestamp')
                        items.append(
                            {
                                'business_id': bid,
                                'title': title,
                                'summary': item_sum,
                                'components': comps,
                                'rating': rating_f,
                                'timestamp': ts,
                            }
                        )
                    self.user2reviews[uid] = items

        # selected triples (support both grouped and single-selection rows)
        triples: List[Tuple[str, str, str]] = []
        open_sel = gzip.open if str(selected_persona_path).endswith('.gz') else open
        with open_sel(selected_persona_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    o = json.loads(s)
                except Exception:
                    continue
                uid = o.get('user_id')
                if not isinstance(uid, str):
                    continue
                selections = o.get('persona_selections')
                if isinstance(selections, list) and selections:
                    for sel in selections:
                        if not isinstance(sel, dict):
                            continue
                        tid = sel.get('target_item_id')
                        spid = sel.get('selected_persona_id')
                        if isinstance(tid, str) and isinstance(spid, str):
                            triples.append((uid, tid, spid))
                else:
                    # single-selection row (4_selected_personas.jsonl)
                    tid = o.get('target_item_id')
                    spid = o.get('selected_persona_id')
                    if isinstance(tid, str) and isinstance(spid, str):
                        triples.append((uid, tid, spid))

        ready: List[ProfileTrainingTuple] = []
        for (uid, target_bid, spid) in triples:
            reviews = self.user2reviews.get(uid) or []
            target_idx = None
            for i, r in enumerate(reviews):
                if r.get('business_id') == target_bid:
                    target_idx = i
            cand = reviews[:target_idx] if target_idx is not None else reviews
            filtered: List[Dict[str, Any]] = []
            for r in cand:
                if r.get('business_id') == target_bid:
                    continue
                rf = r.get('rating')
                if isinstance(rf, float) and rf <= self.min_rating:
                    continue
                filtered.append(r)
            if self.k_recent > 0:
                filtered = filtered[-self.k_recent:]
            rec_texts: List[str] = []
            for r in filtered:
                txt = build_interaction_text(
                    item_summary=r.get('summary'),
                    components=r.get('components'),
                    title=r.get('title'),
                    business_id=r.get('business_id'),
                )
                if txt:
                    rec_texts.append(txt)
            per = self.persona_catalog.get(spid) or {}
            if not per and isinstance(spid, str):
                # Try compose unique id from target business id + short ID (e.g., P3)
                composed = f"{target_bid}_{spid}"
                per = self.persona_catalog.get(composed) or {}
            persona_text = build_persona_text(
                per.get('Name'),
                per.get('Description'),
                per.get('Item_Summary'),
                per.get('Preference_Rationale'),
            )
            if not rec_texts or not persona_text:
                continue
            L = len(rec_texts)
            weights = [self.gamma ** (L - 1 - i) for i in range(L)]
            ready.append(
                ProfileTrainingTuple(
                    user_id=uid,
                    persona_text=persona_text,
                    recent_item_texts=rec_texts,
                    recent_item_weights=weights,
                )
            )
        self._ready = ready

    def __len__(self) -> int:
        return len(self._ready)

    def __getitem__(self, idx: int) -> ProfileTrainingTuple:
        return self._ready[idx]
