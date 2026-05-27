import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphflowdepth.data import RGBDepthDataset
from graphflowdepth.losses import combined_loss
from graphflowdepth.models import GraphFlowDepthModel
from graphflowdepth.utils.checkpointing import load_training_checkpoint, save_training_checkpoint, save_trainable_checkpoint
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
    parser.add_argument("--target-mode", choices=["display_inverse", "auto", "metric"], default="display_inverse")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=DEFAULT_AMP)
    parser.add_argument("--depth-weight", type=float, default=DEFAULT_DEPTH_WEIGHT)
    parser.add_argument("--fm-weight", type=float, default=DEFAULT_FM_WEIGHT)
    parser.add_argument("--edge-weight", type=float, default=DEFAULT_EDGE_WEIGHT)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--val-every-steps", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--latest-checkpoint-name", default="latest.pt")
    parser.add_argument("--best-checkpoint-name", default="best.pt")
    parser.add_argument("--interrupt-checkpoint-name", default="interrupt.pt")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_size = (args.image_height, args.image_width)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_set = RGBDepthDataset(args.data_root, args.train_list, image_size=image_size, target_mode=args.target_mode)
    val_loader = None
    if args.val_list:
        val_set = RGBDepthDataset(args.data_root, args.val_list, image_size=image_size, target_mode=args.target_mode)
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
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    resume_step_in_epoch = 0
    global_step = 0
    best_score = float("inf")
    current_epoch = 1
    current_step = 0
    latest_path = os.path.join(args.checkpoint_dir, args.latest_checkpoint_name)
    best_path = os.path.join(args.checkpoint_dir, args.best_checkpoint_name)
    interrupt_path = os.path.join(args.checkpoint_dir, args.interrupt_checkpoint_name)

    if args.resume:
        info = load_training_checkpoint(model, optimizer, scaler, args.resume, map_location=device)
        payload = info["payload"]
        completed_epoch = bool(payload.get("completed_epoch", False))
        saved_epoch = int(payload.get("epoch", 0))
        start_epoch = saved_epoch + 1 if completed_epoch else max(1, saved_epoch)
        resume_step_in_epoch = 0 if completed_epoch else int(payload.get("step_in_epoch", 0))
        global_step = int(payload.get("global_step", 0))
        best_score = float(payload.get("best_score", best_score))
        print(
            f"resumed from {args.resume}: epoch={start_epoch}, "
            f"skip_steps={resume_step_in_epoch}, global_step={global_step}, best_score={best_score:.6f}"
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            train_loader = make_train_loader(train_set, args, device, epoch)
            current_step = train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                args=args,
                amp_enabled=amp_enabled,
                epoch=epoch,
                resume_step_in_epoch=resume_step_in_epoch if epoch == start_epoch else 0,
                global_step_start=global_step,
                best_score_start=best_score,
                latest_path=latest_path,
                best_path=best_path,
                interrupt_path=interrupt_path,
            )
            global_step = current_step["global_step"]
            best_score = current_step["best_score"]
            current_step_in_epoch = current_step["step_in_epoch"]
            completed_epoch = current_step["completed_epoch"]
            resume_step_in_epoch = 0

            save_full_checkpoint(
                model,
                optimizer,
                scaler,
                latest_path,
                args,
                epoch,
                current_step_in_epoch,
                global_step,
                best_score,
                completed_epoch=completed_epoch,
            )
            print(f"saved latest checkpoint {latest_path}")
            if not completed_epoch:
                print(
                    f"stopped after max_steps_per_epoch={args.max_steps_per_epoch}; "
                    f"resume with --resume {latest_path}"
                )
                break

            if epoch % args.save_every == 0:
                counts = count_parameters(model)
                save_path = os.path.join(args.checkpoint_dir, f"graphflow_epoch_{epoch:03d}.pt")
                save_trainable_checkpoint(
                    model,
                    save_path,
                    extra={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_score": best_score,
                        "param_counts": counts,
                    },
                )
                print(f"saved adapter checkpoint {save_path}")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received; interrupt checkpoint was saved before exit.")
        raise


def make_train_loader(train_set, args, device, epoch):
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    return DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def train_one_epoch(
    model,
    optimizer,
    scaler,
    train_loader,
    val_loader,
    device,
    args,
    amp_enabled,
    epoch,
    resume_step_in_epoch,
    global_step_start,
    best_score_start,
    latest_path,
    best_path,
    interrupt_path,
):
    model.train()
    global_step = global_step_start
    best_score = best_score_start
    current_step = resume_step_in_epoch
    running = {"depth_loss": 0.0, "fm_loss": 0.0, "edge_loss": 0.0, "total_loss": 0.0}
    running_count = 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)

    try:
        for step, batch in enumerate(pbar, start=1):
            current_step = step
            if step <= resume_step_in_epoch:
                continue

            image = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
            global_step += 1
            running_count += 1

            for key in running:
                running[key] += losses[key].detach().item()
            if global_step % args.log_every == 0:
                avg = {key: value / max(running_count, 1) for key, value in running.items()}
                gamma = float(model.gamma.detach().cpu())
                pbar.set_postfix_str(
                    f"tot={avg['total_loss']:.4f} d={avg['depth_loss']:.4f} "
                    f"fm={avg['fm_loss']:.4f} edge={avg['edge_loss']:.4f} gamma={gamma:.5f}"
                )
                tqdm.write(
                    f"global_step={global_step} epoch={epoch} step={step}/{len(train_loader)} "
                    f"total_loss={avg['total_loss']:.6f} depth_loss={avg['depth_loss']:.6f} "
                    f"fm_loss={avg['fm_loss']:.6f} edge_loss={avg['edge_loss']:.6f} gamma={gamma:.6f}"
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_full_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    latest_path,
                    args,
                    epoch,
                    step,
                    global_step,
                    best_score,
                    completed_epoch=False,
                )
                tqdm.write(f"saved latest checkpoint {latest_path} at global_step={global_step}")

            should_validate = val_loader is not None and args.val_every_steps > 0 and global_step % args.val_every_steps == 0
            if should_validate:
                val_losses = validate(model, val_loader, device, args, amp_enabled, max_batches=args.val_batches)
                score = val_losses["total_loss"]
                tqdm.write("val " + " ".join(f"{key}={value:.4f}" for key, value in val_losses.items()))
                if score < best_score:
                    best_score = score
                    save_full_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        best_path,
                        args,
                        epoch,
                        step,
                        global_step,
                        best_score,
                        completed_epoch=False,
                    )
                    tqdm.write(f"saved new best checkpoint {best_path} score={best_score:.6f}")
            elif val_loader is None and losses["total_loss"].detach().item() < best_score:
                best_score = losses["total_loss"].detach().item()
                save_full_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    best_path,
                    args,
                    epoch,
                    step,
                    global_step,
                    best_score,
                    completed_epoch=False,
                )
                tqdm.write(f"saved new best checkpoint {best_path} train_score={best_score:.6f}")
            if args.max_steps_per_epoch is not None and running_count >= args.max_steps_per_epoch:
                return {
                    "global_step": global_step,
                    "best_score": best_score,
                    "step_in_epoch": current_step,
                    "completed_epoch": False,
                }
    except KeyboardInterrupt:
        save_full_checkpoint(
            model,
            optimizer,
            scaler,
            interrupt_path,
            args,
            epoch,
            current_step,
            global_step,
            best_score,
            completed_epoch=False,
        )
        tqdm.write(f"saved interrupt checkpoint {interrupt_path} at global_step={global_step}")
        raise

    return {
        "global_step": global_step,
        "best_score": best_score,
        "step_in_epoch": current_step,
        "completed_epoch": True,
    }


def save_full_checkpoint(
    model,
    optimizer,
    scaler,
    path,
    args,
    epoch,
    step_in_epoch,
    global_step,
    best_score,
    completed_epoch,
):
    save_training_checkpoint(
        model,
        optimizer,
        scaler,
        path,
        extra={
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "best_score": best_score,
            "completed_epoch": completed_epoch,
            "args": vars(args),
            "param_counts": count_parameters(model),
        },
    )


@torch.no_grad()
def validate(model, loader, device, args, amp_enabled, max_batches=None):
    model.eval()
    totals = {"depth_loss": 0.0, "fm_loss": 0.0, "edge_loss": 0.0, "total_loss": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
        if max_batches is not None and batch_idx >= max_batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


if __name__ == "__main__":
    main()
