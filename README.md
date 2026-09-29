# FUSEye: Training-Light Fisheye Detection with Overlapping Views and Zero-Initialized Adapters

Official implementation of **FUSEye**.

FUSEye adapts COCO-pretrained YOLO detectors to fisheye imagery without full-model fine-tuning. It combines **GridViews**, **Z-Adapters**, and **AgreeFusion** to address boundary compression, distortion-induced feature mismatch, and conflicting multi-view detections without camera calibration or dewarping.

!\[FUSEye overview](assets/teaser.png)

## Method

* **GridViews**: processes the full image and four overlapping corner crops.
* **Z-Adapters**: learn lightweight residual feature corrections while the pretrained backbone remains frozen.
* **AgreeFusion**: re-scores and fuses detections supported across multiple views.

The released implementation uses **YOLO26-x**. The detection head and Z-Adapters are trained for fisheye adaptation, followed by a lightweight AgreeFusion scorer.

## Results

Results on the held-out WoodScape MVR split:

|Method|AP50|AP50:95|Car|Person|Bus|
|-|-:|-:|-:|-:|-:|
|Direct transfer|14.80|10.96|15.74|15.67|1.49|
|**FUSEye**|**26.61**|**17.26**|**26.83**|**24.06**|**0.87**|
|Full fine-tuning|31.56|22.82|42.48|22.44|3.54|

FUSEye retains **84.3%** of full-fine-tuning AP50. With only **25%** of the labeled training data, it achieves **25.97 ± 0.34 AP50**, retaining **97.6%** of the full-data result.

## Installation

```bash
conda create -n fuseye python=3.12 -y
conda activate fuseye
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

See `docs/server\_environment.json` for the recorded experiment environment.

## Checkpoints

Place `yolo26x.pt`, `m1\_adapters.pt`, `detect\_head.pt`, and `m3\_v5\_agreement\_scorer\_finalM1.pt` in `weights/`. See `weights/README.md` for checkpoint details.

## Data

Download **WoodScape** under its original dataset terms. Images and annotations are not redistributed.

```text
data/woodscape/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Labels use normalized YOLO format (`class\_id center\_x center\_y width height`) with target IDs `0=car`, `1=person`, and `2=bus`. Training uses FV/RV/MVL; MVR is reserved for evaluation. Exact split manifests are provided in `splits/`.

## Evaluation

```bash
python -m fuseye.cli evaluate \\
  --data data/woodscape \\
  --manifest splits/val\_mvr.json \\
  --base weights/yolo26x.pt \\
  --weights weights \\
  --out runs/evaluation
```

Use `--ablation` for module ablations. See `docs/VALIDATION.md` for validation details.

## Training

Train the adapters and detection head:

```bash
python -m fuseye.train\_detector \\
  --data data/woodscape \\
  --manifest splits/train\_full.json \\
  --base weights/yolo26x.pt \\
  --out runs/full/weights \\
  --epochs 8 --batch-size 8
```

Then collect multi-view candidates and train AgreeFusion:

```bash
python -m fuseye.cli collect \\
  --data data/woodscape --split train \\
  --manifest runs/full/weights/train\_manifest.json \\
  --base weights/yolo26x.pt --weights runs/full/weights \\
  --out runs/full/train\_candidates.json

python -m fuseye.cli train-scorer \\
  --data data/woodscape \\
  --cache runs/full/train\_candidates.json \\
  --out runs/full/weights/m3\_v5\_agreement\_scorer\_finalM1.pt \\
  --epochs 8 --seed 3
```

For the 25% labeled-data experiments, use `splits/train\_fraction25\_seed{0,1,2}.json`. Additional reproduction details are in `docs/REPRODUCIBILITY.md`.

## Contact

For questions about the implementation or reproduction, please open an issue in this repository.

## License

The code is released under [AGPL-3.0](LICENSE). Ultralytics and WoodScape retain their respective licenses and terms.

