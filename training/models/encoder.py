from typing import Dict, Any, Optional, List
import torch


class PersonaEncoder(torch.nn.Module):
    def __init__(
        self,
        st_path: str,
        trust_remote_code: bool = False,
        max_seq_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(st_path, trust_remote_code=trust_remote_code)
        self.max_seq_length = max_seq_length
        try:
            if isinstance(max_seq_length, int) and max_seq_length > 0:
                self.encoder.max_seq_length = max_seq_length
        except Exception:
            pass

    def forward(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        history_texts: Optional[List[str]] = None,
        persona_texts: Optional[List[str]] = None,
    ) -> Any:
        if history_texts is not None and persona_texts is not None:
            all_texts: List[str] = history_texts + persona_texts
            features = self.encoder.tokenize(all_texts)
            if isinstance(self.max_seq_length, int) and self.max_seq_length > 0:
                try:
                    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                        if key in features and isinstance(features[key], torch.Tensor):
                            features[key] = features[key][:, : self.max_seq_length]
                except Exception:
                    pass
            device = next(self.encoder.parameters()).device
            for k in list(features.keys()):
                v = features[k]
                if isinstance(v, torch.Tensor):
                    features[k] = v.to(device)
            out = self.encoder(features)
            token_embeddings = out.get('token_embeddings') if isinstance(out, dict) else out[0]
            cls_embeddings = token_embeddings[:, 0]
            all_emb = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)
            num_hist = len(history_texts)
            return all_emb[:num_hist], all_emb[num_hist:]

        if inputs is None:
            raise ValueError('Either inputs or (history_texts, persona_texts) must be provided')
        texts = inputs['texts']
        features = self.encoder.tokenize(texts)
        device = next(self.encoder.parameters()).device
        for k in list(features.keys()):
            v = features[k]
            if isinstance(v, torch.Tensor):
                features[k] = v.to(device)
        out = self.encoder(features)
        token_embeddings = out.get('token_embeddings') if isinstance(out, dict) else out[0]
        cls_embeddings = token_embeddings[:, 0]
        emb = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)
        return { 'emb': emb }





