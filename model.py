"""Build YOLO26-x with the original checkpoint-compatible adapter layout."""
from pathlib import Path
import torch
from ultralytics import YOLO
from .adapters import GeometricAdapterV2, AdapterWrapper

INDICES = [6, 8, 10]
HEAD_INDEX = 23

def build_model(base, device="cuda:0", weights=None):
    if not Path(base).is_file():
        raise FileNotFoundError(f"Supply an existing official YOLO26-x checkpoint: {base}")
    model = YOLO(str(base))
    model.to(device)
    model.model.eval()
    seq = model.model.model
    if len(seq) != 24 or seq[HEAD_INDEX].nc != 80:
        raise ValueError("This release targets the 80-class COCO YOLO26-x architecture only")
    acts = {}
    handles = [seq[i].register_forward_hook(
        lambda mod, inp, out, idx=i: acts.__setitem__(idx, out.shape[1])) for i in INDICES]
    with torch.no_grad():
        model.model(torch.zeros(1, 3, 640, 640, device=device))
    for h in handles:
        h.remove()
    adapters = {}
    for i in INDICES:
        adapters[i] = GeometricAdapterV2(acts[i]).to(device)
        seq[i] = AdapterWrapper(seq[i], adapters[i])
    head = seq[HEAD_INDEX]
    if sum(p.numel() for a in adapters.values() for p in a.parameters()) != 222480:
        raise ValueError("Adapter widths do not match the final YOLO26-x checkpoint")
    if weights is not None:
        weights = Path(weights)
        state = torch.load(weights / "m1_adapters.pt", map_location=device, weights_only=True)
        for i, adapter in adapters.items():
            adapter.load_state_dict(state[f"adapter_{i}"])
        head.load_state_dict(torch.load(weights / "detect_head.pt", map_location=device, weights_only=True))
    model.model.eval()
    return model, adapters, head
