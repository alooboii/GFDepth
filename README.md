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

List files are plain text. Each non-comment line contains:

```text
relative/or/absolute/rgb_path relative/or/absolute/depth_path
```

Depth can be an image or `.npy` file. Image depths with values above 255 are interpreted as millimeters and divided by 1000.

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
  --image-width 518
```

The script prints total, frozen, trainable, and trainable-percent parameter counts at startup. Checkpoints save trainable adapter weights by default.

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

Evaluation reports AbsRel, RMSE, delta1, and patch edge-gradient MAE. Visualizations include RGB, GT depth, baseline frozen DA2 depth, GraphFlow-adapted depth, absolute error, and velocity magnitude maps.

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
