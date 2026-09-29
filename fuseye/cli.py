"""FUSEye inference, candidate collection, scorer training, and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from . import fusion
from .data import select_stems
from .evaluation import evaluate
from .inference import collect, predict_views
from .model import build_model


def save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scorer(weights,device):
    scorer=fusion.ClusterScorerV2(12).to(device)
    scorer.load_state_dict(torch.load(Path(weights)/'m3_v5_agreement_scorer_finalM1.pt',map_location=device,weights_only=True))
    return scorer.eval()


def fused(records,scorer):
    return [{'file':r['file'],'preds':fusion.gate_preds(c,scorer)}
            for r,c in zip(records,fusion.build_clusters(records))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('collect','evaluate','predict'):
        p=sub.add_parser(name)
        p.add_argument('--base',required=True)
        p.add_argument('--weights',type=Path,default=Path('weights'))
        p.add_argument('--device',default='cuda:0')
        p.add_argument('--out',type=Path,required=True)
        if name != 'predict':
            p.add_argument('--data',type=Path,required=True)
            p.add_argument('--manifest',type=Path)
            p.add_argument('--limit',type=int)
        if name == 'collect':
            p.add_argument('--split',choices=['train','val'],required=True)
        if name == 'evaluate':
            p.add_argument('--ablation',action='store_true',help='Evaluate all eight module combinations')
        if name == 'predict':
            p.add_argument('--image',type=Path,required=True)
            p.add_argument('--draw-threshold',type=float,default=0.25,help='Visualization only; raw JSON retains all predictions')
    p=sub.add_parser('train-scorer')
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--seed',type=int,default=3)
    a=parser.parse_args()
    fusion.DEVICE=a.device
    if a.command == 'train-scorer':
        if a.out.exists():
            raise FileExistsError(a.out)
        cache=json.loads(a.cache.read_text())
        if cache.get('split') != 'train':
            raise ValueError('Scorer must be trained on training candidates, never validation candidates')
        gt=fusion.load_gt(a.data/'labels'/'train')
        records=cache['records']
        if not records or len({r['file'] for r in records}) != len(records) or any(r['file'] not in gt for r in records):
            raise ValueError('Missing or duplicate training image records/labels')
        X,y,_=fusion.make_dataset(records,gt)
        if len(X)==0 or not np.any(y>0.5):
            raise ValueError('No positive training clusters')
        cls=X[:,-3:].copy()
        model=fusion.train_rank_scorer(X,y,cls,epochs=a.epochs,seed=a.seed)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),a.out)
        save(a.out.with_suffix('.json'),{'epochs':a.epochs,'seed':a.seed,'images':len(records),'clusters':len(X),
                                      'positive_clusters':int(y.sum()),'cache_sha256':digest(a.cache)})
        return
    model,_,_=build_model(a.base,a.device,a.weights)
    if a.command == 'predict':
        scorer=load_scorer(a.weights,a.device)
        with Image.open(a.image) as src:
            img=src.convert('RGB')
        records=[{'file':a.image.stem,'dets':predict_views(model,img)}]
        predictions=fused(records,scorer)
        a.out.mkdir(parents=True,exist_ok=True)
        save(a.out/'predictions.json',predictions)
        draw=ImageDraw.Draw(img)
        for x,y,w,h,score,c in predictions[0]['preds']:
            if score<a.draw_threshold:
                continue
            box=((x-w/2)*img.width,(y-h/2)*img.height,(x+w/2)*img.width,(y+h/2)*img.height)
            draw.rectangle(box,outline='red',width=3)
            draw.text((box[0],box[1]),f'{fusion.WS_NAMES[c]} {score:.2f}',fill='red')
        img.save(a.out/'prediction.jpg')
        return
    split=a.split if a.command=='collect' else 'val'
    stems=select_stems(a.data,split,a.manifest)
    if a.limit is not None:
        if a.limit < 1:
            raise ValueError('limit must be positive')
        stems=stems[:a.limit]
    records=collect(model,a.data,split,stems)
    if a.command == 'collect':
        if a.out.exists():
            raise FileExistsError('Choose a new candidate cache path')
        save(a.out,{'split':split,'protocol':'FUSEye-final-5view-conf002','base_sha256':digest(a.base),
                    'adapter_sha256':digest(a.weights/'m1_adapters.pt'),'head_sha256':digest(a.weights/'detect_head.pt'),
                    'records':records})
        return
    scorer=load_scorer(a.weights,a.device)
    a.out.mkdir(parents=True,exist_ok=True)
    predictions=fused(records,scorer)
    save(a.out/'predictions.json',predictions)
    results={'M1+M2+M3':evaluate(a.data,predictions)}
    if a.ablation:
        del model
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
        from ultralytics import YOLO
        base=YOLO(a.base).to(a.device)
        original=collect(base,a.data,'val',stems)
        for tag,pv in [('baseline',original),('M1',records),('M2',original),('M1+M2',records),('M3',original),('M1+M3',records),('M2+M3',original)]:
            if 'M2' not in tag:
                pv=[{'file':r['file'],'dets':{'full':r['dets']['full']}} for r in pv]
            if 'M3' in tag:
                pr=fused(pv,scorer)
            else:
                pr=[{'file':r['file'],'preds':fusion.baseline_nms(r['dets'],0.25)} for r in pv]
            results[tag]=evaluate(a.data,pr)
    full_stems={p.stem for p in (a.data/'labels'/'val').glob('*.txt')}
    save(a.out/'metrics.json',{'scope':'full validation' if set(stems)==full_stems else 'subset','results':results})
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
