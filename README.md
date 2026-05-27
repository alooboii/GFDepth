# GraphFlowDepth

GraphFlowDepth is a research prototype for testing a lightweight trainable Patch Graph Flow adapter inside a frozen pretrained Depth Anything V2 depth pipeline.

This is not a new full depth foundation model. The intended experiment is low-compute and parameter-efficient: keep the Depth Anything V2-Small encoder and DPT depth head frozen, insert a small trainable adapter on the deepest patch feature map, and measure whether local patch-token transition velocities improve depth predictions.

## Core Idea

For an RGB image, frozen DA2 produces intermediate features:

```text
Z1, Z2, Z3, Z4 = frozen_DA2_encoder_intermediate_features(x)
M = GraphFlowAdapter(Z4)
Z4_tilde = Z4 + gamma * M
D_hat = frozen_DA2_DPT_head(Z1, Z2, Z3, Z4_tilde)
```

Only these parts train:

- GraphFlow adapter
- down projection, velocity aggregation projection, up projection
- scalar residual gate `gamma`
- auxiliary edge-depth head

DA2 encoder and DA2 DPT/depth head remain frozen. `gamma` is initialized to `0.0`, so the initial adapted model is exactly the frozen DA2 baseline.

## No Config System

There is no Hydra, no Lightning, no WandB, and no YAML. Training and evaluation use plain PyTorch plus `argparse`. `train.py` also has a clearly marked manual defaults section near the top for quick notebook/Kaggle-style edits.

## MVP Loss

```text
L = 1.0 * L_depth + 0.05 * L_FM + 0.1 * L_edge
```

- `L_depth`: masked L1 depth loss on valid pixels
- `L_FM`: MSE between predicted transition velocity and stop-gradient projected feature difference
- `L_edge`: L1 loss on patch-level depth gradients for R, D, DR, and DL directed edges

Optional SILog and boundary-head scaffolding are present, but the boundary head is not enabled by default.

## Data Format

List files may be plain text or CSV. Plain text rows contain:

```text
relative/or/absolute/rgb_path relative/or/absolute/depth_path
```

CSV rows contain:

```text
relative/or/absolute/rgb_path,relative/or/absolute/depth_path
```

The Kaggle NYU Depth V2 dataset works directly:

```bash
--data-root /kaggle/input/nyu-depth-v2/nyu_data \
--train-list /kaggle/input/nyu-depth-v2/nyu_data/data/nyu2_train.csv
```

The default target mode is `display_inverse`, which converts every target to normalized DA2-style inverse depth in `[0, 1]`: closer is brighter and farther is darker. For NYU's 8-bit grayscale targets this is `1 - gray / 255`. For 16-bit/metric targets, the loader normalizes valid metric depth by image percentiles and then inverts it. This keeps training, validation, notebook plots, and saved visualizations on one convention.

Other modes are available for debugging:

- `--target-mode auto`: preserves the older loader behavior.
- `--target-mode metric`: uses metric-style loading where possible.

## Depth Anything V2 Setup

Install the official Depth Anything V2 code so this import works:

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

Point the wrapper at the DA2-Small weights with:

```bash
export DEPTH_ANYTHING_V2_SMALL_CKPT=/path/to/depth_anything_v2_vits.pth
```

If your local DA2 fork exposes different internals, adjust only `graphflowdepth/models/da2_wrapper.py`. The adapter and losses are independent of DA2 internals.

## Train

```bash
python train.py \
  --data-root /path/to/data \
  --train-list train.txt \
  --val-list val.txt \
  --checkpoint-dir checkpoints \
  --batch-size 4 \
  --image-height 518 \
  --image-width 518 \
  --target-mode display_inverse
```

The script prints total, frozen, trainable, and trainable-percent parameter counts at startup. Checkpoints save trainable adapter weights by default.

For large datasets, use step-based checkpointing and resume:

```bash
python train.py \
  --data-root /path/to/data \
  --train-list train.csv \
  --val-list val.csv \
  --checkpoint-dir checkpoints \
  --batch-size 2 \
  --epochs 3 \
  --save-every-steps 500 \
  --val-every-steps 1000 \
  --val-batches 50
```

This overwrites `checkpoints/latest.pt` every `--save-every-steps` and overwrites `checkpoints/best.pt` whenever the validation slice improves. To stop after a manageable slice and inspect results:

```bash
python train.py ... --max-steps-per-epoch 3000
```

Resume full training state, including optimizer, AMP scaler, epoch, step, and best score:

```bash
python train.py ... --resume checkpoints/latest.pt
```

If training is interrupted with Ctrl-C, `checkpoints/interrupt.pt` is saved before exit and can also be passed to `--resume`.

## Evaluate

```bash
python eval.py \
  --data-root /path/to/data \
  --val-list val.txt \
  --checkpoint checkpoints/graphflow_epoch_020.pt \
  --batch-size 4 \
  --save-visuals \
  --visual-dir visuals
```

For the default `--target-mode display_inverse`, evaluation reports display-depth MAE/RMSE, epsilon-masked AbsRel/delta1 diagnostics, and patch edge-gradient MAE. Standard AbsRel is not a good headline metric for normalized inverse depth because valid far pixels can be close to zero. In `auto` or `metric` target modes, evaluation falls back to the standard AbsRel/RMSE/delta1 metrics. Visualizations include RGB, GT depth, baseline frozen DA2 depth, GraphFlow-adapted depth, absolute error, and velocity magnitude maps.

## Repository Layout

```text
graphflowdepth/
  models/
    da2_wrapper.py
    graphflow_adapter.py
    edge_heads.py
    graphflow_depth_model.py
  losses/
    depth_losses.py
    graphflow_losses.py
  data/
    datasets.py
    transforms.py
  utils/
    metrics.py
    param_count.py
    checkpointing.py
    visualization.py
train.py
eval.py
README.md
requirements.txt
```
