"""COCO evaluation in the original 640x640 coordinate convention."""
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .data import labels
from .fusion import WS_NAMES

def evaluate(data, predictions):
    if not predictions or len({p['file'] for p in predictions}) != len(predictions):
        raise ValueError('Predictions must contain unique, non-empty image records')
    gt = {'images':[],'annotations':[],'categories':[{'id':c,'name':n} for c,n in WS_NAMES.items()],'info':{}}
    dt=[]
    for iid,record in enumerate(predictions,1):
        stem = record['file']
        gt['images'].append({'id':iid,'file_name':stem,'width':640,'height':640})
        for c,x,y,w,h in labels(Path(data)/'labels'/'val'/(stem+'.txt')):
            gt['annotations'].append({'id':len(gt['annotations'])+1,'image_id':iid,'category_id':c,
                'bbox':[(x-w/2)*640,(y-h/2)*640,w*640,h*640],'area':w*h*640*640,'iscrowd':0})
        for x,y,w,h,score,c in record['preds']:
            dt.append({'image_id':iid,'category_id':int(c),'bbox':[(x-w/2)*640,(y-h/2)*640,w*640,h*640],'score':score})
    coco=COCO(); coco.dataset=gt; coco.createIndex()
    if dt:
        detections=coco.loadRes(dt)
    else:
        detections=COCO(); detections.dataset={**gt,'annotations':[]}; detections.createIndex()
    ev=COCOeval(coco,detections,'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    per_class={}
    for i,c in enumerate(ev.params.catIds):
        p=ev.eval['precision'][:,:,i,0,-1]; p=p[p>-1]
        per_class[WS_NAMES[c]]=round(float(p.mean()),4) if len(p) else None
    return {'mAP50':round(float(ev.stats[1]),4),'mAP50-95':round(float(ev.stats[0]),4),
            'per_class':per_class,'dets':len(dt),'images':len(predictions)}
