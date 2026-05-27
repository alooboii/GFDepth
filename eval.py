import argparse
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphflowdepth.data import RGBDepthDataset
from graphflowdepth.models import GraphFlowDepthModel
from graphflowdepth.utils.checkpointing import load_trainable_checkpoint
from graphflowdepth.utils.metrics import depth_metrics, display_depth_metrics, patch_edge_gradient_error
from graphflowdepth.utils.visualization import save_depth_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GraphFlowDepth adapter.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="depth_anything_v2_small")
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=518)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--target-mode", choices=["display_inverse", "auto", "metric"], default="display_inverse")
    parser.add_argument("--output-calibration", dest="output_calibration", action="store_true")
    parser.add_argument("--no-output-calibration", dest="output_calibration", action="store_false")
    parser.set_defaults(output_calibration=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--visual-dir", default="visuals")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset = RGBDepthDataset(
        args.data_root,
        args.val_list,
        image_size=(args.image_height, args.image_width),
        target_mode=args.target_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = GraphFlowDepthModel(
        backbone=args.backbone,
        graph_dim=args.graph_dim,
        output_calibration=args.output_calibration,
    ).to(device)
    load_info = load_trainable_checkpoint(model, args.checkpoint, map_location=device)
    if load_info["unexpected"]:
        print(f"unexpected checkpoint keys: {load_info['unexpected']}")
    model.eval()

    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval")):
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            valid = batch["valid_mask"].to(device)
            pred, aux = model(image, depth_gt=depth, valid_mask=valid, return_baseline=args.save_visuals)
            if args.target_mode == "display_inverse":
                metrics = display_depth_metrics(pred, depth, valid)
            else:
                metrics = depth_metrics(pred, depth, valid)
            patch_hw = next(iter(aux["velocities"].values())).shape[-2:]
            metrics["patch_edge_grad_mae"] = patch_edge_gradient_error(pred, depth, valid, patch_hw)
            for key, value in metrics.items():
                totals[key] += value
            count += 1

            if args.save_visuals:
                for i in range(image.shape[0]):
                    stem = os.path.splitext(os.path.basename(batch["rgb_path"][i]))[0]
                    velocity_maps = {name: value[i] for name, value in aux["velocities"].items()}
                    save_depth_visualization(
                        args.visual_dir,
                        f"{batch_idx:04d}_{stem}",
                        image[i],
                        depth[i],
                        aux.get("baseline_depth", None)[i] if "baseline_depth" in aux else None,
                        pred[i],
                        velocity_maps=velocity_maps,
                    )

    print("Evaluation metrics")
    for key, value in totals.items():
        print(f"{key}: {value / max(count, 1):.6f}")


if __name__ == "__main__":
    main()
