import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphflowdepth.data import RGBDepthDataset
from graphflowdepth.losses import combined_loss
from graphflowdepth.models import GraphFlowDepthModel
from graphflowdepth.utils.checkpointing import save_trainable_checkpoint
from graphflowdepth.utils.param_count import count_parameters, format_parameter_report

# -------------------------
# Manual defaults
# -------------------------
DEFAULT_BACKBONE = "depth_anything_v2_small"
DEFAULT_GRAPH_DIM = 64
DEFAULT_DIRECTIONS = ["R", "D", "DR", "DL"]
DEFAULT_DIRECTION_EMBED_DIM = 16
DEFAULT_ALPHA_EMBED_DIM = 16
DEFAULT_FLOW_HIDDEN_DIM = 128
DEFAULT_GAMMA_INIT = 0.0

DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_AMP = True

DEFAULT_DEPTH_WEIGHT = 1.0
DEFAULT_FM_WEIGHT = 0.05
DEFAULT_EDGE_WEIGHT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphFlowDepth adapter with frozen DA2.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--graph-dim", type=int, default=DEFAULT_GRAPH_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=518)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=DEFAULT_AMP)
    parser.add_argument("--depth-weight", type=float, default=DEFAULT_DEPTH_WEIGHT)
    parser.add_argument("--fm-weight", type=float, default=DEFAULT_FM_WEIGHT)
    parser.add_argument("--edge-weight", type=float, default=DEFAULT_EDGE_WEIGHT)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_size = (args.image_height, args.image_width)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_set = RGBDepthDataset(args.data_root, args.train_list, image_size=image_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if args.val_list:
        val_set = RGBDepthDataset(args.data_root, args.val_list, image_size=image_size)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = GraphFlowDepthModel(
        backbone=args.backbone,
        graph_dim=args.graph_dim,
        directions=DEFAULT_DIRECTIONS,
        direction_embed_dim=DEFAULT_DIRECTION_EMBED_DIM,
        alpha_embed_dim=DEFAULT_ALPHA_EMBED_DIM,
        flow_hidden_dim=DEFAULT_FLOW_HIDDEN_DIM,
        gamma_init=DEFAULT_GAMMA_INIT,
    ).to(device)
    model.assert_da2_frozen()
    print(format_parameter_report(model))

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp_enabled = args.amp and device.type == "cuda"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"depth_loss": 0.0, "fm_loss": 0.0, "edge_loss": 0.0, "total_loss": 0.0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_depth, aux = model(image, depth_gt=depth, valid_mask=valid)
                losses = combined_loss(
                    pred_depth,
                    depth,
                    valid,
                    aux,
                    depth_weight=args.depth_weight,
                    fm_weight=args.fm_weight,
                    edge_weight=args.edge_weight,
                )

            scaler.scale(losses["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            for key in running:
                running[key] += losses[key].detach().item()
            if step % args.log_every == 0:
                avg = {key: value / step for key, value in running.items()}
                pbar.set_postfix({key: f"{value:.4f}" for key, value in avg.items()})

        if val_loader is not None:
            val_losses = validate(model, val_loader, device, args, amp_enabled)
            print("val " + " ".join(f"{key}={value:.4f}" for key, value in val_losses.items()))

        if epoch % args.save_every == 0:
            counts = count_parameters(model)
            save_path = os.path.join(args.checkpoint_dir, f"graphflow_epoch_{epoch:03d}.pt")
            save_trainable_checkpoint(model, save_path, extra={"epoch": epoch, "param_counts": counts})
            print(f"saved {save_path}")


@torch.no_grad()
def validate(model, loader, device, args, amp_enabled):
    model.eval()
    totals = {"depth_loss": 0.0, "fm_loss": 0.0, "edge_loss": 0.0, "total_loss": 0.0}
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_depth, aux = model(image, depth_gt=depth, valid_mask=valid)
            losses = combined_loss(
                pred_depth,
                depth,
                valid,
                aux,
                depth_weight=args.depth_weight,
                fm_weight=args.fm_weight,
                edge_weight=args.edge_weight,
            )
        for key in totals:
            totals[key] += losses[key].item()
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


if __name__ == "__main__":
    main()
