"""Prepare the camera split from existing YOLO-format WoodScape images/labels.

This command does not download or redistribute WoodScape and does not guess
the conversion from an unknown raw annotation format.
"""
import argparse
import json
from pathlib import Path
import shutil
from .data import image_path, labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True,help='Flat directory of WoodScape images')
    p.add_argument('--labels',type=Path,required=True,help='Flat YOLO labels, car=0 person=1 bus=2')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--copy',action='store_true',help='Copy images; default creates symlinks')
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new dataset destination')
    pending=[]
    for f in sorted(a.labels.glob('*.txt')):
        camera=f.stem.rsplit('_',1)[-1]
        if camera not in ('FV','RV','MVL','MVR'):
            raise ValueError(f'Unrecognized camera suffix in {f.name}')
        rows=labels(f)
        source=image_path(a.images,f.stem)
        pending.append((f,source,'val' if camera=='MVR' else 'train',bool(rows)))
    if not pending:
        raise ValueError('No labels found')
    for split in ('train','val'):
        for kind in ('images','labels'):
            (a.out/kind/split).mkdir(parents=True)
    for label,image,split,_ in pending:
        shutil.copy2(label,a.out/'labels'/split/label.name)
        dest=a.out/'images'/split/image.name
        if a.copy:
            shutil.copy2(image,dest)
        else:
            dest.symlink_to(image.resolve())
    summary={split:{'images':sum(s==split for _,_,s,_ in pending),
                    'target_images':sum(s==split and target for _,_,s,target in pending)} for split in ('train','val')}
    (a.out/'split_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
