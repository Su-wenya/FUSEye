"""Train the final Z-Adapters + existing COCO detection head recipe."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WoodScapeDataset, collate, select_stems
from .model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--target-only', action='store_true')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-batches', type=int, help='Smoke test only; not a paper run')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    if any((a.out/n).exists() for n in ('m1_adapters.pt','detect_head.pt')):
        raise FileExistsError('Choose a new output directory to preserve existing checkpoints')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    stems = select_stems(a.data, 'train', a.manifest, a.target_only)
    ds = WoodScapeDataset(a.data, stems)
    dl = DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,
                    collate_fn=collate,pin_memory=a.device.startswith('cuda'))
    model, adapters, head = build_model(a.base,a.device)
    if isinstance(model.model.args,dict):
        model.model.args = SimpleNamespace(**model.model.args)
    for k,v in {'box':7.5,'cls':0.5,'dfl':1.5}.items():
        if not hasattr(model.model.args,k):
            setattr(model.model.args,k,v)
    for param in model.model.parameters():
        param.requires_grad_(False)
    ap = [p for ad in adapters.values() for p in ad.parameters()]
    hp = list(head.parameters())
    for param in ap+hp:
        param.requires_grad_(True)
    # Match the historical warm-up: eval backbone, newly inserted train-mode adapters.
    for adapter in adapters.values():
        adapter.train()
    with torch.no_grad():
        model.model(torch.zeros(1,3,640,640,device=a.device))
    opt = torch.optim.Adam([{'params':ap,'lr':1e-4},{'params':hp,'lr':5e-5}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs*len(dl))
    history=[]
    print(f'Training images={len(stems)}, adapters={sum(p.numel() for p in ap)}, head={sum(p.numel() for p in hp)}',flush=True)
    for epoch in range(a.epochs):
        # Preserve the original training behavior: parameters outside adapters/head
        # are frozen, but BatchNorm running statistics update in train mode.
        model.model.train()
        losses=[]
        for index,batch in enumerate(dl):
            if a.max_batches is not None and index >= a.max_batches:
                break
            batch = {k:v.to(a.device) for k,v in batch.items()}
            opt.zero_grad()
            loss_items,_ = model.model(batch)
            loss = loss_items.sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ap+hp,max_norm=10.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        if not losses:
            raise ValueError('No training batches')
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
        print(history[-1],flush=True)
    torch.save({f'adapter_{i}':ad.state_dict() for i,ad in adapters.items()},a.out/'m1_adapters.pt')
    torch.save(head.state_dict(),a.out/'detect_head.pt')
    metadata={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'stems':stems,'history':history,'n_images':len(stems),
              'note':'Only adapter and detection-head state is exported, matching the historical inference reconstruction.'}
    (a.out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (a.out/'train_manifest.json').write_text(json.dumps({'stems':stems},indent=2)+'\n')


if __name__ == '__main__':
    main()
