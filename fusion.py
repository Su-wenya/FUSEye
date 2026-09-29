"""Final GridViews and AgreeFusion algorithms extracted from the experiment code.

Coordinates are normalized (cx, cy, width, height); class IDs are car=0,
person=1, bus=2. Device is set explicitly by each command-line entry point.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

DEVICE = "cpu"
WS_IDS = [0, 1, 2]
WS_NAMES = {0: "car", 1: "person", 2: "bus"}
COCO_TO_WS = {0: 1, 2: 0, 5: 2}
VIEW_NAMES = ["full", "tl", "tr", "bl", "br"]
EPOCHS, BS, TAU_SINGLE = 8, 2048, 0.25

def xyxy(d):
    xc, yc, w, h = d[0], d[1], d[2], d[3]
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def greedy_clusters(dets, iou_m):
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    used = [False] * len(dets)
    boxes = [xyxy(d) for d in dets]
    clusters = []
    for i in range(len(dets)):
        if used[i]:
            continue
        used[i] = True
        cl = [dets[i]]
        for j in range(i + 1, len(dets)):
            if not used[j] and iou(boxes[i], boxes[j]) >= iou_m:
                used[j] = True
                cl.append(dets[j])
        clusters.append(cl)
    return clusters


def final_nms(preds, iou_thresh=0.5):
    if not preds:
        return []
    boxes = torch.tensor(
        [[p[0] - p[2] / 2, p[1] - p[3] / 2, p[0] + p[2] / 2, p[1] + p[3] / 2]
         for p in preds],
        device=DEVICE,
    )
    scores = torch.tensor([p[4] for p in preds], device=DEVICE)
    classes = torch.tensor([p[5] for p in preds], device=DEVICE)
    keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=DEVICE)
    for c in classes.unique():
        mask = classes == c
        if mask.sum() == 0:
            continue
        keep = nms(boxes[mask], scores[mask], iou_thresh)
        keep_mask[torch.where(mask)[0][keep]] = True
    return [p for p, k in zip(preds, keep_mask.cpu().tolist()) if k]


def baseline_nms(img_dets, conf_min):
    all_dets = []
    for vi, vn in enumerate(VIEW_NAMES):
        for d in img_dets.get(vn, []):
            if d[4] >= conf_min:
                all_dets.append(d + [vi])
    preds = []
    for c in WS_IDS:
        cd = [d for d in all_dets if d[5] == c]
        if cd:
            preds.extend([d[:4] + [d[4], c] for d in cd])
    return final_nms(preds)
def center_disp(cl):
    cxs = [d[0] for d in cl]
    cys = [d[1] for d in cl]
    ws = [d[2] for d in cl]
    hs = [d[3] for d in cl]
    mcx = sum(cxs) / len(cxs)
    mcy = sum(cys) / len(cys)
    disp = math.sqrt(
        sum((cx - mcx) ** 2 + (cy - mcy) ** 2 for cx, cy in zip(cxs, cys))
        / len(cxs)
    )
    norm = max(sum(ws) / len(ws), sum(hs) / len(hs), 1e-6)
    return disp / norm


def mean_pair_iou(cl):
    boxes = [xyxy(d) for d in cl]
    total = 0.0
    cnt = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            total += iou(boxes[i], boxes[j])
            cnt += 1
    return total / cnt if cnt else 1.0


def cluster_features(cl):
    confs = [d[4] for d in cl]
    k = len({d[6] for d in cl})
    c_max = max(confs)
    c_mean = sum(confs) / len(confs)
    c_min = min(confs)
    or_score = 1.0
    for ci in confs:
        or_score *= 1.0 - ci
    or_score = 1.0 - or_score
    disp = center_disp(cl)
    m_iou = mean_pair_iou(cl)
    w = sum(d[2] for d in cl) / len(cl)
    h = sum(d[3] for d in cl) / len(cl)
    cls = cl[0][5]
    onehot = [0.0, 0.0, 0.0]
    onehot[cls] = 1.0
    return (
        [k / 5.0, c_max, c_mean, c_min, or_score, min(disp, 2.0) / 2.0, m_iou, w, h]
        + onehot
    )


def build_clusters(per_view, iou_m=0.5):
    out = []
    for item in per_view:
        img_dets = item["dets"]
        all_dets = []
        for vi, vn in enumerate(VIEW_NAMES):
            for d in img_dets.get(vn, []):
                all_dets.append(d + [vi])
        clusters = []
        for c in WS_IDS:
            cd = [d for d in all_dets if d[5] == c]
            if cd:
                clusters.extend(greedy_clusters(cd, iou_m))
        out.append(clusters)
    return out


def load_gt(lbl_dir):
    files = sorted(lbl_dir.glob("*.txt"))
    gt = {}
    for f in files:
        stem = f.stem
        boxes = []
        for line in open(f):
            p = line.strip().split()
            if len(p) >= 5 and int(float(p[0])) in WS_IDS:
                cls = int(float(p[0]))
                cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                boxes.append([cls, cx, cy, w, h])
        gt[stem] = boxes
    return gt


def label_clusters(clusters, gt_boxes):
    labels = []
    used = [False] * len(gt_boxes)
    for cl in clusters:
        best = max(cl, key=lambda d: d[4])
        pb = (
            best[0] - best[2] / 2,
            best[1] - best[3] / 2,
            best[0] + best[2] / 2,
            best[1] + best[3] / 2,
        )
        tp = 0
        for gi, g in enumerate(gt_boxes):
            if used[gi] or g[0] != best[5]:
                continue
            gb = (g[1] - g[3] / 2, g[2] - g[4] / 2, g[1] + g[3] / 2, g[2] + g[4] / 2)
            if iou(pb, gb) >= 0.5:
                used[gi] = True
                tp = 1
                break
        labels.append(tp)
    return labels


def make_dataset(per_view, gt, iou_m=0.5):
    clusters_all = build_clusters(per_view, iou_m)
    X = []
    y = []
    c_max_all = []
    for item, clusters in zip(per_view, clusters_all):
        lbls = label_clusters(clusters, gt.get(item["file"], []))
        for cl, lb in zip(clusters, lbls):
            X.append(cluster_features(cl))
            y.append(lb)
            c_max_all.append(max(d[4] for d in cl))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(c_max_all)
class ClusterScorerV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_rank_scorer(X, y, cls, epochs=EPOCHS, bs=BS, seed=3):
    torch.manual_seed(seed)
    model = ClusterScorerV2(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    ct = torch.from_numpy(cls).to(DEVICE)

    pos = [0, 0, 0]
    neg = [0, 0, 0]
    for i in range(len(y)):
        c = int(cls[i].argmax())
        if y[i] > 0.5:
            pos[c] += 1
        else:
            neg[c] += 1
    print(f"class positives: {pos}, negatives: {neg}", flush=True)
    base_w = [max(1.0, neg[c] / max(pos[c], 1)) for c in range(3)]
    print(f"class pos_weight (BCE): {[round(w, 2) for w in base_w]}", flush=True)

    n = len(y)
    for e in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        nb = 0
        for s in range(0, n, bs):
            ids = perm[s : s + bs]
            bx = Xt[ids]
            by = yt[ids]
            bc = ct[ids].argmax(1)
            logits = model(bx)
            pw = torch.tensor(
                [base_w[c] for c in bc.cpu().tolist()], dtype=torch.float32, device=DEVICE
            )
            loss = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pw)

            pidx = (by > 0.5).nonzero(as_tuple=True)[0]
            nidx = (by <= 0.5).nonzero(as_tuple=True)[0]
            if len(pidx) >= 2 and len(nidx) >= 2:
                k = min(256, len(pidx), len(nidx))
                pi = pidx[torch.randperm(len(pidx), device=DEVICE)[:k]]
                ni = nidx[torch.randperm(len(nidx), device=DEVICE)[:k]]
                rank_loss = F.softplus(-(logits[pi] - logits[ni])).mean()
                loss = loss + 0.5 * rank_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(ids)
            nb += 1
        print(f"  epoch {e + 1}: loss={total / max(nb * bs, 1):.4f}", flush=True)
    return model


def scorer_prob(model, cl):
    f = torch.tensor([cluster_features(cl)], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(f)).item()


def gate_preds(clusters, model, tau_single=TAU_SINGLE):
    preds = []
    for cl in clusters:
        best = max(cl, key=lambda d: d[4])
        n_views = len({d[6] for d in cl})
        if n_views < 2:
            if best[4] >= tau_single:
                preds.append(list(best[:4]) + [best[4], best[5]])
            continue
        p = scorer_prob(model, cl)
        score = max(best[4], p)
        disp = center_disp(cl)
        if disp <= 0.6:
            wsum = sum(d[4] for d in cl)
            box = [
                sum(d[0] * d[4] for d in cl) / wsum,
                sum(d[1] * d[4] for d in cl) / wsum,
                sum(d[2] * d[4] for d in cl) / wsum,
                sum(d[3] * d[4] for d in cl) / wsum,
            ]
        else:
            box = best[:4]
        preds.append(box + [score, best[5]])
    return final_nms(preds)
