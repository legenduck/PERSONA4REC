import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from typing import List, Optional
import os

from .datasets.yelp import ProfileTrainingTuple


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        mask_same_user: bool = False,
        mask_percpos: bool = False,
        percpos_threshold: float = 0.95,
        debug_print_texts: bool = False,
        debug_metrics: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        # Allow disabling AMP via env
        self.use_amp = bool(int(os.environ.get('USE_AMP', '1'))) and use_amp
        self.device = next(model.parameters()).device

        # Learned scale (kept but not applied initially)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * 0.0)
        self.logit_scale.to(self.device)
        # Temperature factor for logits (default 0.7, can set via LOGIT_TEMP)
        try:
            self.logit_temp = float(os.environ.get('LOGIT_TEMP', '0.7'))
        except Exception:
            self.logit_temp = 0.7

        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        self.accumulated_profile_embeddings: List[torch.Tensor] = []
        self.accumulated_persona_embeddings: List[torch.Tensor] = []
        self.accumulated_user_ids: List[str] = []

        self.mask_same_user = bool(mask_same_user)
        self.mask_percpos = bool(mask_percpos)
        self.percpos_threshold = float(percpos_threshold)
        self.debug_print_texts = bool(debug_print_texts)
        self.debug_metrics = bool(debug_metrics)
        self._current_step_idx: int = -1

    def step(self, batch: List[ProfileTrainingTuple], step_idx: int, grad_accum_steps: int) -> Optional[torch.Tensor]:
        flat_history_texts: List[str] = []
        seg_lengths: List[int] = []
        per_sample_weights: List[List[float]] = []
        persona_texts: List[str] = []
        local_user_ids = [inst.user_id for inst in batch]

        for inst in batch:
            persona_texts.append(inst.persona_text)
            seg_lengths.append(len(inst.recent_item_texts))
            per_sample_weights.append(list(inst.recent_item_weights))
            flat_history_texts.extend(inst.recent_item_texts)

        filtered_indices = [i for i, l in enumerate(seg_lengths) if l > 0]
        if not filtered_indices:
            return None
        persona_texts = [persona_texts[i] for i in filtered_indices]
        per_sample_weights = [per_sample_weights[i] for i in filtered_indices]
        seg_lengths = [seg_lengths[i] for i in filtered_indices]
        local_user_ids = [local_user_ids[i] for i in filtered_indices]

        with autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
            hist_emb, per_emb = self.model(history_texts=flat_history_texts, persona_texts=persona_texts)

            profiles: List[torch.Tensor] = []
            start = 0
            for seg_len, w_list in zip(seg_lengths, per_sample_weights):
                end = start + seg_len
                seg = hist_emb[start:end]
                start = end
                if seg.numel() == 0:
                    continue
                w = torch.tensor(w_list, device=seg.device, dtype=seg.dtype)
                denom = torch.clamp(w.sum(), min=1e-6)
                prof = (seg * (w/denom).view(-1, 1)).sum(dim=0)
                profiles.append(prof)
            prof_emb = torch.stack(profiles, dim=0)

            if ((step_idx + 1) % max(int(grad_accum_steps), 1)) != 0:
                self.accumulated_profile_embeddings.append(prof_emb.detach().cpu())
                self.accumulated_persona_embeddings.append(per_emb.detach().cpu())
                self.accumulated_user_ids.extend(local_user_ids)
                return None

        prev_prof = torch.cat(self.accumulated_profile_embeddings, dim=0).to(self.device) if self.accumulated_profile_embeddings else None
        prev_pers = torch.cat(self.accumulated_persona_embeddings, dim=0).to(self.device) if self.accumulated_persona_embeddings else None
        prev_ids = list(self.accumulated_user_ids)
        if prev_prof is not None:
            prof_local = torch.cat([prev_prof, prof_emb], dim=0)
            pers_local = torch.cat([prev_pers, per_emb], dim=0)
            ids_local = prev_ids + local_user_ids
        else:
            prof_local = prof_emb
            pers_local = per_emb
            ids_local = local_user_ids

        self.accumulated_profile_embeddings = []
        self.accumulated_persona_embeddings = []
        self.accumulated_user_ids = []

        self._current_step_idx = int(step_idx)
        loss = self._loss_allgather(prof_local, pers_local, ids_local)
        # Clamp logit scale to avoid explosion (kept small)
        with torch.no_grad():
            self.logit_scale.data.clamp_(-2.0, 2.0)
        return loss

    def _loss_allgather(self, prof_local: torch.Tensor, pers_local: torch.Tensor, ids_local: list) -> torch.Tensor:
        if dist.is_initialized() and dist.get_world_size() > 1:
            def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
                world = dist.get_world_size()
                buff = [torch.zeros_like(t) for _ in range(world)]
                dist.all_gather(buff, t)
                buff[dist.get_rank()] = t
                return torch.cat(buff, dim=0)

            g_prof = all_gather_tensor(prof_local)
            g_pers = all_gather_tensor(pers_local)
            world = dist.get_world_size()
            gathered_ids = [None for _ in range(world)]
            dist.all_gather_object(gathered_ids, ids_local)
            global_user_ids = [u for sub in gathered_ids for u in sub]
        else:
            g_prof = prof_local
            g_pers = pers_local
            global_user_ids = ids_local

        g_prof_n = F.normalize(g_prof.float(), dim=-1, eps=1e-12)
        g_pers_n = F.normalize(g_pers.float(), dim=-1, eps=1e-12)
        raw_sim = torch.matmul(g_prof_n, g_pers_n.T)
        # Temperature-only scaling for stability (logit_scale disabled initially)
        logits = raw_sim * self.logit_temp
        B = logits.shape[0]
        labels = torch.arange(B, device=self.device)

        final_mask = torch.ones_like(logits, dtype=torch.bool)
        if self.mask_same_user:
            same_user_mask = torch.ones_like(logits, dtype=torch.bool)
            for i in range(B):
                for j in range(B):
                    if i != j and global_user_ids[i] == global_user_ids[j]:
                        same_user_mask[i, j] = False
            final_mask = final_mask & same_user_mask
        if self.mask_percpos:
            pos_scores = torch.diag(raw_sim).detach()
            max_neg = pos_scores * self.percpos_threshold
            hard_neg = raw_sim > max_neg.unsqueeze(1)
            try:
                hard_neg.fill_diagonal_(False)
            except Exception:
                pass
            final_mask = final_mask & (~hard_neg)

        # Preserve diagonal (positive) always
        eye = torch.eye(B, dtype=torch.bool, device=final_mask.device)
        final_mask = final_mask | eye
        # Guarantee at least 1 negative per row
        valid_neg = (final_mask & (~eye)).sum(dim=1)
        need_unmask = valid_neg == 0
        if need_unmask.any():
            raw_no_diag = raw_sim.masked_fill(eye, -1e9)
            top_idx = raw_no_diag.argmax(dim=1)
            final_mask[torch.arange(B, device=final_mask.device), top_idx] = True

        # Use a large negative constant to avoid -inf
        logits = logits.masked_fill(~final_mask, -1e9)
        loss = F.cross_entropy(logits, labels, reduction='mean', label_smoothing=0.05)
        return loss
