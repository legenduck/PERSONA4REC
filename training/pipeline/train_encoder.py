#!/usr/bin/env python3
import argparse
import os
import sys
import json
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure repo root is on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Local imports
from training.models.encoder import PersonaEncoder
from training.models.trainer import Trainer
from training.models.datasets.yelp import YelpPersonaDataset
from training.models.datasets.amazon import AmazonPersonaDataset


def init_distributed() -> int:
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def build_dataloader(dataset, batch_size: int, world_size: int, local_rank: int, num_workers: int = 0):
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda x: x,
            persistent_workers=num_workers > 0,
        )
        return dl, sampler
    else:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda x: x,
            persistent_workers=num_workers > 0,
        )
        return dl, None


def main():
    ap = argparse.ArgumentParser(description='Train encoder (Yelp/Amazon) inside repository (no subprocess).')
    ap.add_argument('--dataset', choices=['yelp', 'amazon'], required=True)
    ap.add_argument('--data_cfg', required=True)
    ap.add_argument('--train_cfg', required=True)
    args = ap.parse_args()

    local_rank = init_distributed()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    with open(args.data_cfg, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_cfg, 'r', encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)

    # Resolve paths relative to repo root
    def rp(p):
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        return os.path.join(_REPO_ROOT, p)

    # Read required config keys (strict schema)
    selected_cfg = data_cfg.get('selected')
    persona_cfg = data_cfg.get('persona')
    history_cfg = data_cfg.get('loo_history')

    # Validate presence
    missing_keys = []
    if not isinstance(selected_cfg, str):
        missing_keys.append('selected')
    if not isinstance(persona_cfg, str):
        missing_keys.append('persona')
    if not isinstance(history_cfg, str):
        missing_keys.append('loo_history')
    if missing_keys:
        raise ValueError(f"Missing required data config keys: {', '.join(missing_keys)}")

    selected = rp(selected_cfg)
    catalog = rp(persona_cfg)
    history = rp(history_cfg)

    # Validate file existence to give early, clear errors
    missing_paths = [p for p in [selected, catalog, history] if not os.path.exists(p)]
    if missing_paths:
        raise FileNotFoundError(f"Data files not found: {missing_paths}.\nResolved relative to repo root: {_REPO_ROOT}")

    # Dataset switch (Amazon to be added)
    if args.dataset == 'yelp':
        dataset = YelpPersonaDataset(
            selected_persona_path=selected,
            persona_catalog_path=catalog,
            loo_history_path=history,
            k_recent=int(train_cfg.get('k_recent', 10)),
            gamma=float(train_cfg.get('gamma', 0.85)),
            min_rating=float(train_cfg.get('min_rating', 3.0)),
            debug=False,
        )
    elif args.dataset == 'amazon':
        dataset = AmazonPersonaDataset(
            selected_persona_path=selected,
            persona_catalog_path=catalog,
            loo_history_path=history,
            k_recent=int(train_cfg.get('k_recent', 10)),
            gamma=float(train_cfg.get('gamma', 0.85)),
            min_rating=float(train_cfg.get('min_rating', 3.0)),
            debug=False,
        )
    else:
        raise NotImplementedError('Unknown dataset')

    dataloader, sampler = build_dataloader(
        dataset=dataset,
        batch_size=int(train_cfg.get('batch_size', 8)),
        world_size=world_size,
        local_rank=local_rank,
        num_workers=int(train_cfg.get('num_workers', 0)),
    )

    if (world_size == 1) or (not dist.is_initialized()) or (dist.get_rank() == 0):
        effective_batch = int(train_cfg.get('batch_size', 8)) * max(world_size, 1) * max(int(train_cfg.get('grad_accum_steps', 1)), 1)
        print(json.dumps({
            'dataset_size': len(dataset),
            'steps_per_epoch': len(dataloader),
            'batch_size_per_gpu': int(train_cfg.get('batch_size', 8)),
            'world_size': world_size,
            'grad_accum_steps': int(train_cfg.get('grad_accum_steps', 1)),
            'effective_global_batch': effective_batch,
            'num_workers': int(train_cfg.get('num_workers', 0)),
            'epochs': int(train_cfg.get('epochs', 1)),
        }))

    model_core = PersonaEncoder(
        st_path=str(train_cfg.get('st_path', 'BAAI/bge-m3')),
        trust_remote_code=False,
        max_seq_length=int(train_cfg.get('max_len', 768)),
    ).to(device)
    try:
        model_core.train()
    except Exception:
        pass

    if world_size > 1:
        model = DDP(
            model_core,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            static_graph=False,
        )
    else:
        model = model_core

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg.get('lr', 1e-5)))
    trainer = Trainer(
        model,
        optimizer,
        use_amp=True,
        mask_same_user=bool(train_cfg.get('mask_same_user', False)),
        mask_percpos=bool(train_cfg.get('mask_percpos', False)),
        percpos_threshold=float(train_cfg.get('percpos_threshold', 0.95)),
        debug_print_texts=False,
        debug_metrics=True,
    )
    # Optionally exclude logit_scale from early updates
    freeze_steps = int(train_cfg.get('freeze_logit_scale_steps', 0))

    try:
        from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
        num_training_steps = len(dataloader) * int(train_cfg.get('epochs', 1)) // max(int(train_cfg.get('grad_accum_steps', 1)), 1)
        if str(train_cfg.get('scheduler', 'linear')) == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(train_cfg.get('warmup_steps', 100)), num_training_steps=num_training_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(train_cfg.get('warmup_steps', 100)), num_training_steps=num_training_steps
            )
    except Exception:
        scheduler = None

    total_steps = 0
    # Honor max_steps from config or env
    max_steps_cfg = train_cfg.get('max_steps', None)
    max_steps_env = os.environ.get('MAX_STEPS', None)
    max_steps = None
    if isinstance(max_steps_cfg, int):
        max_steps = max_steps_cfg
    elif max_steps_env is not None:
        try:
            max_steps = int(max_steps_env)
        except Exception:
            max_steps = None

    epochs = int(train_cfg.get('epochs', 1))
    grad_accum = int(train_cfg.get('grad_accum_steps', 1))

    for ep in range(epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        for step, batch in enumerate(dataloader):
            if not batch:
                continue
            # Freeze logit_scale for first N steps if configured
            if freeze_steps and total_steps < freeze_steps:
                trainer.logit_scale.requires_grad = False
            else:
                trainer.logit_scale.requires_grad = True

            loss = trainer.step(batch, step_idx=step, grad_accum_steps=grad_accum)
            if loss is not None:
                scaled_loss = loss / max(int(grad_accum), 1)
                if isinstance(model, DDP):
                    trainer.scaler.scale(scaled_loss).backward()
                    trainer.scaler.step(optimizer)
                    trainer.scaler.update()
                else:
                    trainer.scaler.scale(scaled_loss).backward()
                    trainer.scaler.step(optimizer)
                    trainer.scaler.update()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                optimizer.zero_grad(set_to_none=True)
                if (world_size == 1) or (not dist.is_initialized()) or (dist.get_rank() == 0):
                    try:
                        loss_val = float(loss.detach().cpu().item())
                    except Exception:
                        loss_val = None
                    print(json.dumps({'epoch': ep, 'step': f"{step}/{len(dataloader)}", 'global_step': total_steps + 1, 'loss': loss_val}))
                total_steps += 1
            if isinstance(max_steps, int) and total_steps >= max_steps:
                break
        if isinstance(max_steps, int) and total_steps >= max_steps:
            break

    if dist.is_initialized():
        dist.barrier()

    if (world_size == 1) or (not dist.is_initialized()) or (dist.get_rank() == 0):
        out_root = rp(data_cfg.get('out_root', f'training/outputs/{args.dataset}'))
        os.makedirs(out_root, exist_ok=True)
        out_dir = os.path.join(out_root, 'train')
        os.makedirs(out_dir, exist_ok=True)
        try:
            # If wrapped by DDP, unwrap
            core = model.module if isinstance(model, DDP) else model
            core.save_pretrained(out_dir)
        except Exception:
            try:
                (model.module.encoder if isinstance(model, DDP) else model.encoder).save(out_dir)
            except Exception:
                pass
        try:
            with open(os.path.join(out_dir, 'trainer_aux.json'), 'w') as f:
                json.dump({'logit_scale': float(trainer.logit_scale.detach().cpu().item())}, f)
        except Exception:
            pass
        print(json.dumps({'saved_to': out_dir}))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
