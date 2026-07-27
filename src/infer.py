
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_dataloaders, list_sample_ids, NucleiDataset, train_val_split
from model import UNet
from metrics import dice_coefficient, iou_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data/stage1_train")
    p.add_argument("--checkpoint", type=str, default="../outputs/best_model.pt")
    p.add_argument("--output-dir", type=str, default="../outputs")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet(in_channels=3, out_channels=1, base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    img_size = ckpt["img_size"]

    ids = list_sample_ids(args.data_root)
    if args.max_samples is not None:
        import random
        random.Random(args.seed).shuffle(ids)
        ids = ids[:args.max_samples]
    _, val_ids = train_val_split(ids, args.val_fraction, args.seed)
    val_ids = val_ids[:args.n_samples]

    ds = NucleiDataset(args.data_root, val_ids, img_size=img_size, augment=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(val_ids), 4, figsize=(14, 3.2 * len(val_ids)))
    if len(val_ids) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground truth mask", "Predicted mask", "Overlay (GT=green, Pred=red)"]

    for row, (img_t, mask_t) in enumerate(ds):
        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob > args.threshold).astype(np.uint8)

        d = dice_coefficient(torch.sigmoid(logits), mask_t.unsqueeze(0).to(device)).item()
        i = iou_score(torch.sigmoid(logits), mask_t.unsqueeze(0).to(device)).item()

        img_np = img_t.permute(1, 2, 0).numpy()
        gt_np = mask_t[0].numpy()

        overlay = img_np.copy()
        overlay[..., 1] = np.where(gt_np > 0, 1.0, overlay[..., 1])   # green = GT
        overlay[..., 0] = np.where(pred > 0, 1.0, overlay[..., 0])    # red = prediction

        axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_np, cmap="gray"); axes[row, 1].axis("off")
        axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].axis("off")
        axes[row, 3].imshow(overlay); axes[row, 3].axis("off")
        axes[row, 3].set_title(f"Dice={d:.3f}  IoU={i:.3f}", fontsize=9)

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[row, c].set_title(title if c != 3 else axes[row, c].get_title(), fontsize=10)

    plt.tight_layout()
    save_path = out_dir / "qualitative_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved qualitative results to {save_path}")


if __name__ == "__main__":
    main()
