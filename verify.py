"""Verify the four released checkpoints against their recorded SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=Path('weights'))
    a=p.parse_args()
    expected=json.loads((a.weights/'SHA256.json').read_text())
    failed=[]
    for name,value in expected.items():
        path=a.weights/name
        if not path.is_file():
            failed.append(f'{name}: missing')
            continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                h.update(block)
        if h.hexdigest()!=value:
            failed.append(f'{name}: SHA-256 mismatch')
        else:
            print(f'OK {name}')
    if failed:
        raise SystemExit('\n'.join(failed))


if __name__=='__main__':
    main()
