# Release validation

Validation date: 2026-09-29. Commands ran in a separate temporary workspace
using the existing server environment and original dataset. Original experiment
scripts, checkpoints and result files were not overwritten.

## Full fresh inference

The released inference pipeline was run on **all 2,145 MVR validation images**
from image files, using the four historical checkpoints. This was not only
a re-evaluation of a saved detection cache.

| Metric | Fresh result | Historical final result |
|---|---:|---:|
| AP50 | 0.2661 | 0.2661 |
| AP50:95 | 0.1726 | 0.1726 |
| Car AP50:95 | 0.2683 | 0.2683 |
| Person AP50:95 | 0.2406 | 0.2406 |
| Bus AP50:95 | 0.0087 | 0.0087 |
| Final detections | 38,491 | 38,491 |

These values match at the original four-decimal reporting precision. Evidence
is in `results/validated_fresh_full.json`.

## Additional checks

- Exact identity of a newly zero-initialized adapter.
- Original-source versus release feature/clustering/gating outputs for the
  first 50 historical validation records.
- Re-evaluation of all historical candidate records reproduces all headline
  and per-class results, as well as the final detection count.
- Fresh predictions on the first three images have matching shapes and maximum
  absolute numeric differences no larger than 5.97e-8 from historical outputs.
- All eight ablation command branches execute on a three-image subset.
- Detector training runs one optimization batch on three actual images,
  including one image with no target labels; the loss is finite and checkpoint
  export succeeds.
- Candidate generation runs on 20 training images. The scorer completes one
  training epoch and exports its checkpoint.
- The newly trained detector checkpoint can be reloaded for inference.
- The single-image prediction command saves JSON and an annotated image.
- The data-preparation command splits a minimal FV/MVR input correctly.
- All four downloaded checkpoint files match their recorded SHA-256 values.
- Ultralytics Python files match their installed distribution RECORD hashes.
- All released Python files compile.

Detailed evidence is in `results/validated_algorithms.json` and
`results/validated_entrypoints.json`.

## Limits of validation

This work did not rerun the full eight-epoch detector training or all
limited-label and cross-model experiments. Historical full-data detector
training had no fixed seed. A one-batch training smoke test verifies executable
training, gradients and checkpoint I/O; it does not establish convergence or
bit-identical retraining. Installation into a completely clean machine was
not performed. Data preparation was tested with existing YOLO-format labels,
not a verified raw annotation converter.
