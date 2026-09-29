# Reproduction notes

## Version identification

The final implementation was identified by the matching complete ablation
table, not by timestamps or a filename containing "final".

| Component | Original source | Release implementation |
|---|---|---|
| M1 / Z-Adapters | `modules.py`, `train_m1_head_ft_v2.py` | `adapters.py`, `model.py`, `train_detector.py` |
| M2 / GridViews | `m3_fusion_v4.py` | `inference.py` |
| M3 / AgreeFusion | `m3_curriculum.py`, `m3_v5_agreement_scorer.py` | `fusion.py` |
| Final protocol | `final_m3_v5_ablation.py` | `cli.py`, `evaluation.py` |

`source_manifest.json` records SHA-256 values of the original source files.
Old tangent-plane generation and feature-level cross-projection gates found in
`modules.py` are not part of this final method and have not been included.
Although feature functions came from a file named `m3_curriculum.py`, the final
scorer does **not** use curriculum training.

## Differences between manuscript wording and executable protocol

1. **Training population.** The final full-data log records 6,089 training
   images, including seven without car/person/bus labels. There are 6,082
   target-containing images. The original full-data loader includes all 6,089.
   The manuscript describes filtering to 6,082. The default release preserves
   the actual 6,089-image behavior. `--target-only` changes the experiment and
   should not be described as an exact reconstruction of the released weights.
   The 25% experiments selected 1,520 images from the 6,082 eligible images;
   their original three manifests are included.

2. **Frozen weights versus BatchNorm state.** Original training calls
   `model.train()` on the entire detector. Non-adapter/non-head parameters have
   `requires_grad=False`, but BatchNorm running statistics still update.
   Only adapter/head states are saved. The final inference loader reconstructs
   the detector from the original COCO checkpoint and replaces those saved
   states, restoring the original remaining buffers. This release preserves
   that behavior. A claim that all backbone state stays unchanged throughout
   training would be inaccurate. Evaluate the reconstructed checkpoint, not
   the still-in-memory model immediately after training.

3. **Trainable parameters.** The adapters have 222,480 parameters, the
   detection head 6,440,952, and the scorer 4,385. Total parameters optimized
   across the two stages are 6,667,817. Newly introduced parameters are 226,865.

4. **Baseline protocol.** The saved ablation baseline is AP50=0.1462 and
   AP50:95=0.1105. The manuscript main table uses 0.1480 and 0.1096 from a
   different direct-transfer evaluation. The release reports the ablation
   protocol baseline; it does not silently substitute the main-table value.

5. **M3 behavior.** Candidates use per-view confidence 0.02. Class-wise greedy
   clustering uses IoU 0.5 and compares each new box to the cluster seed, not
   to a moving mean. Multiple boxes from the same view can enter a cluster;
   the gate counts distinct view IDs. Single-view clusters need confidence
   0.25. Multi-view scores are `max(best_confidence, sigmoid(MLP(features)))`.
   The score does not impose a separate positive/negative acceptance threshold
   on multi-view clusters. Box averaging uses raw confidence weights when
   normalized center dispersion is at most 0.6. Final class-wise NMS uses 0.5.

6. **Scorer supervision.** A cluster is positive when its highest-confidence
   box matches an unused same-class training GT box at IoU >= 0.5. Matching
   follows the original cluster and annotation order. Features and labels are
   generated from the detector's training set, not from held-out calibration
   images. The final scorer uses class-weighted BCE plus a batch-wide pairwise
   ranking term with coefficient 0.5; pairs are not restricted to the same class.

7. **Image and metric conventions.** Source images are 1280x966. Each crop is
   stretched to 640x640 using PIL bilinear resize. This is not letterboxing or
   geometric rectification. Evaluation scales normalized coordinates to a
   640x640 canvas and uses COCO's default maxDets=100. Area-specific COCO
   summaries consequently refer to this resized canvas. The paper's reported
   headline metrics are overall AP and per-class AP50:95.

8. **Randomness.** The original full-data detector training script did not fix
   a random seed. The release supplies `--seed` for future traceability but
   does not promise bit-identical retraining of the historical checkpoint.
   The original scorer training fixes the torch seed to 3. Use the supplied
   checkpoints for direct metric reproduction.

## Release changes

- Removed absolute server paths and import-time log writes.
- Preserved checkpoint keys and the feature/scoring/clustering functions.
- Added commands for candidate generation, scorer training, inference and
  evaluation, with explicit paths and data manifests.
- Added validation for empty/missing labels, missing images and incompatible
  image sizes. Empty-label tensors have shape `(0,4)` so batches remain valid.
- JSON candidate files include source checkpoint hashes and split identity.
  No untrusted pickle file is required by the released command-line tools.
- Dataset images/annotations, historical caches, obsolete experiments, private
  server details and credentials are not included.

## Scope

This release targets the requested final YOLO26-x + M1 + M2 + M3 experiment.
It includes the principal ablation and the original 25% split manifests.
It does not package the full-fine-tuning baselines, other detector families,
or every exploratory experiment. Historical results are labeled as reported
results; fresh validation evidence is stored separately.

The available working dataset was already YOLO-formatted. A raw-WoodScape
annotation conversion matching this dataset has not been established from the
selected final scripts. `prepare_data` therefore accepts existing YOLO labels
with explicitly documented IDs; it does not pretend to reproduce an unknown
raw-data conversion. Exact reproduction requires matching the provided
filename manifests and label semantics.
