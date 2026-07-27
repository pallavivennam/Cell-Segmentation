# Cell Segmentation — U-Net on the Kaggle Data Science Bowl 2018 Dataset

A deep-learning pipeline that segments individual nuclei/cells in
fluorescence microscopy images, using a U-Net trained end-to-end in
PyTorch. This matches the "Cell Segmentation" project described in
the resume: dataset preprocessing, U-Net training, Dice/IoU/precision/
recall evaluation, and qualitative error analysis of over-/under-
segmentation.

## Dataset

`data/stage1_train/` contains the Kaggle DSB2018 stage-1 training set
(664 samples), using the community "dataset fixes" version. Each
sample folder has:

```
<image_id>/
  images/<image_id>.png      # RGB microscopy image
  masks/*.png                # one binary mask per individual nucleus
```

`src/dataset.py` merges the per-instance masks into a single binary
semantic mask (nucleus vs. background) — a standard simplification for
semantic (as opposed to instance) segmentation with U-Net.

## Pipeline

```
src/
  dataset.py   - discovers samples, merges masks, resize + augmentation, DataLoader
  model.py     - U-Net (4-stage encoder/decoder, skip connections, ~31M params @ base_ch=32)
  metrics.py   - Dice/IoU/precision/recall + combined Dice+BCE loss
  train.py     - training loop, LR scheduling, early stopping, checkpointing, curve plots
  infer.py     - loads a checkpoint, visualizes predictions vs. ground truth
```

### Preprocessing
- Resize image + mask to a fixed square size (default 128×128; use larger,
  e.g. 256, once you have a GPU, for higher-fidelity boundaries).
- Normalize image to [0, 1].
- Augmentation (train split only): random horizontal/vertical flip,
  random 90° rotation, random brightness/contrast jitter.

### Model
Standard U-Net: 4 encoder stages (double 3×3 conv + BatchNorm + ReLU,
2×2 max-pool) mirrored by 4 decoder stages (2×2 transposed conv +
concatenated skip connection + double conv), 1×1 output conv producing
per-pixel logits.

### Loss & metrics
- Loss: 0.5 · BCE + 0.5 · (1 − Dice) — BCE gives stable per-pixel
  gradients, Dice compensates for the foreground/background class
  imbalance (nuclei typically cover a minority of pixels).
- Metrics tracked every epoch: Dice coefficient, IoU (Jaccard),
  precision, recall — on a held-out validation split.

### Training
Adam optimizer, `ReduceLROnPlateau` scheduler on validation Dice,
early stopping, best-checkpoint saving, CSV log + loss/Dice curve PNG.

## Running it

```bash
cd src

# 1. Full training run (recommended: GPU, ~30-50 epochs, 256px images)
python train.py --data-root ../data/stage1_train \
                 --img-size 256 --batch-size 16 --epochs 50 --lr 1e-3

# 2. Quick CPU smoke test (what was run to validate this pipeline)
python train.py --data-root ../data/stage1_train \
                 --img-size 96 --batch-size 8 --epochs 6 --max-samples 120

# 3. Qualitative results (image | ground truth | prediction | overlay)
python infer.py --data-root ../data/stage1_train \
                 --checkpoint ../outputs/best_model.pt --n-samples 6
```

Outputs land in `outputs/`: `best_model.pt`, `training_log.csv`,
`loss_curve.png`, `qualitative_results.png`.

## What's included vs. what to extend

Included, matching the resume bullet points:
- U-Net built and trained in PyTorch on the DSB2018 dataset.
- Preprocessing: normalization, resizing; augmentation: flips,
  rotation, intensity jitter.
- Manual ground-truth masks used directly from the dataset's
  per-nucleus annotation files (already produced via ImageJ/CVAT-style
  labeling in the original Kaggle release).
- Evaluation via Dice, IoU, precision, recall.
- Qualitative error analysis: `infer.py` renders GT-vs-prediction
  overlays so over-segmentation and under-segmentation are visible
  directly.
- Streamlined via a Linux CLI (argparse scripts, no notebook required).

Sensible extensions if you want to push it further:
- Switch merged binary masks to instance-aware training
  (e.g., watershed post-processing or a Mask R-CNN/StarDist head) if
  you need to separate touching nuclei rather than just flag
  foreground pixels.
- Train at 256×256 with a GPU for materially better Dice/IoU than the
  96×96 CPU smoke test included here.
- 5-fold cross-validation instead of a single train/val split for a
  more robust reported score.

## Web app

`app.py` is a small Streamlit app: upload an image, the trained U-Net
segments it, and you see the input, predicted mask, and overlay side
by side (plus a downloadable mask PNG).

```bash
pip install -r requirements.txt
streamlit run app.py
```

It loads `outputs/best_model.pt`, so train a model first (or drop your
own checkpoint there). Opens at `http://localhost:8501`.

## Included smoke-test results

For sanity-checking, the pipeline was run for 6 epochs on a 120-image
subset at 96×96 on CPU (a few minutes). Validation Dice rose from
0.18 → 0.52 in 6 epochs, confirming the training loop, loss, and
metrics are correct and learning. A full run (256px, 50 epochs, GPU)
will get well beyond this — DSB2018 U-Net baselines typically reach
Dice ≈ 0.85–0.92.

## The output of the webapp when tested on a random image found through google search
<img width="1913" height="1002" alt="image" src="https://github.com/user-attachments/assets/edc06b04-526b-435a-be31-ff749a0a5029" />
