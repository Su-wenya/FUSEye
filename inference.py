"""Fixed five-view inference used by the final YOLO26 experiments."""
from PIL import Image
import torch
from .fusion import COCO_TO_WS
from .data import image_path

def grid_views(width=1280, height=966):
    pw, ph = int(width*(1-0.33)), int(height*(1-0.33))
    sx, sy = width-pw, height-ph
    return [('full',0,0,width,height),('tl',0,0,pw,ph),
            ('tr',sx,0,width,ph),('bl',0,sy,pw,height),('br',sx,sy,width,height)]

def predict_views(model, image, single=False):
    if image.size != (1280,966):
        raise ValueError(f"Paper protocol requires 1280x966 images, got {image.size}")
    views = grid_views()[:1] if single else grid_views()
    result = {}
    for name,x1,y1,x2,y2 in views:
        patch = image.crop((x1,y1,x2,y2)).resize((640,640),Image.Resampling.BILINEAR)
        with torch.no_grad():
            r = model(patch, conf=0.02, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for box, confidence, cid in zip(r.boxes.xywhn.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.cpu().tolist()):
                if int(cid) not in COCO_TO_WS:
                    continue
                cx,cy,w,h = box
                if name != 'full':
                    cx,cy,w,h = (x1+cx*(x2-x1))/1280,(y1+cy*(y2-y1))/966,w*(x2-x1)/1280,h*(y2-y1)/966
                    if not (0 <= cx <= 1 and 0 <= cy <= 1):
                        continue
                dets.append([cx,cy,w,h,confidence,COCO_TO_WS[int(cid)]])
        result[name] = dets
    return result

def collect(model, data, split, stems, single=False):
    records = []
    for i,stem in enumerate(stems):
        with Image.open(image_path(data/'images'/split,stem)) as img:
            views = predict_views(model,img.convert('RGB'),single)
        records.append({'file':stem,'dets':views})
        if (i+1)%100 == 0:
            print(f"inference {i+1}/{len(stems)}",flush=True)
    return records
