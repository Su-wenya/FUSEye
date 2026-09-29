"""YOLO-format WoodScape data validation and split selection."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

WS2COCO = {0: 2, 1: 0, 2: 5}
EXTENSIONS = (".jpg", ".webp", ".png", ".jpeg")

def image_path(root, stem):
    found = [Path(root)/(stem+ext) for ext in EXTENSIONS if (Path(root)/(stem+ext)).is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one image for {stem}, found {len(found)} in {root}")
    return found[0]

def labels(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected class cx cy w h in {path}")
        c, x, y, w, h = map(float, fields)
        if c != int(c) or not np.isfinite([c,x,y,w,h]).all() or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized detection label in {path}: {line}")
        if int(c) in WS2COCO:
            rows.append((int(c), x,y,w,h))
    return rows

def select_stems(data, split, manifest=None, target_only=False):
    data = Path(data)
    available = sorted(p.stem for p in (data/'labels'/split).glob('*.txt'))
    if not available:
        raise ValueError(f"No labels found for {split} in {data}")
    stems = available
    if manifest:
        obj = json.loads(Path(manifest).read_text())
        stems = obj['stems']
        if len(stems) != len(set(stems)) or not set(stems).issubset(available):
            raise ValueError("Split manifest contains duplicates or missing labels")
    if target_only:
        stems = [s for s in stems if labels(data/'labels'/split/(s+'.txt'))]
    for s in stems:
        image_path(data/'images'/split, s)
    return stems

class WoodScapeDataset(Dataset):
    def __init__(self, data, stems):
        self.data, self.stems = Path(data), stems
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, idx):
        stem = self.stems[idx]
        with Image.open(image_path(self.data/'images'/'train', stem)) as src:
            img = np.array(src.convert('RGB').resize((640,640),Image.Resampling.BILINEAR))
        rows = labels(self.data/'labels'/'train'/(stem+'.txt'))
        return {'img': torch.from_numpy(img).permute(2,0,1).float()/255,
                'cls': torch.tensor([WS2COCO[r[0]] for r in rows],dtype=torch.long),
                'bbox': torch.tensor([r[1:] for r in rows],dtype=torch.float32).reshape(-1,4)}

def collate(batch):
    return {'img': torch.stack([b['img'] for b in batch]),
            'cls': torch.cat([b['cls'] for b in batch]),
            'bboxes': torch.cat([b['bbox'] for b in batch]),
            'batch_idx': torch.cat([torch.full((len(b['cls']),),i,dtype=torch.long) for i,b in enumerate(batch)])}
