# Checkpoints

The accompanying `FUSEye-weights.zip` contains all four checkpoints identified
from the final experiment. Unzip it into this directory:

| Filename | Contents |
|---|---|
| `yolo26x.pt` | Original COCO-pretrained base used in the experiment |
| `m1_adapters.pt` | Three Z-Adapters at layers 6, 8 and 10 |
| `detect_head.pt` | Adapted 80-class COCO detection head, layer 23 |
| `m3_v5_agreement_scorer_finalM1.pt` | Final AgreeFusion MLP |

The original COCO-pretrained `yolo26x.pt` has its required SHA-256
is recorded in `SHA256.json`, together with the other three checkpoint hashes.
The detection head keeps all 80 COCO outputs; final predictions retain only
car, person and bus. Loading it into a new three-class head is not compatible.

The weights are ignored by Git. For public distribution, attach the weights
archive to a GitHub Release and link it from the main README. No public download
URL is claimed here because no release has been uploaded.
