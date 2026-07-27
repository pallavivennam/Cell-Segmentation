import argparse
import csv
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import make_dataloaders
from model import UNet
from metrics import DiceBCELoss, evaluate, dice_coefficient


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for nucleus segmentation")
    p.add_argument("--data-root", type=str, default="../data/stage1_train")
    p.add_argument("--output-dir", type=str, default="../outputs")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-samples", type=int, default=None,
                    help="Subsample dataset for a quick smoke test")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8,
                    help="Early-stopping patience (epochs without val Dice improvement)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_ids, val_ids = make_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train samples: {len(train_ids)} | Val samples: {len(val_ids)}")

    model = UNet(in_channels=3, out_channels=1, base_ch=args.base_ch).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_dice = -1.0
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                batch_dice = dice_coefficient(probs, masks).item()

            running_loss += loss.item()
            running_dice += batch_dice
            n_batches += 1

        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["dice"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
              f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} | "
              f"{elapsed:.1f}s")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": args.img_size,
                "base_ch": args.base_ch,
                "val_dice": best_dice,
            }, out_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val Dice improvement for {args.patience} epochs.")
                break

    # Write CSV log
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    # Plot loss / dice curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x = [r["epoch"] for r in log_rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs_x, [r["train_loss"] for r in log_rows], label="train loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")

        axes[1].plot(epochs_x, [r["train_dice"] for r in log_rows], label="train Dice")
        axes[1].plot(epochs_x, [r["val_dice"] for r in log_rows], label="val Dice")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice"); axes[1].legend(); axes[1].set_title("Dice coefficient")

        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        print(f"Saved training curves to {out_dir / 'loss_curve.png'}")
    except ImportError:
        print("matplotlib not available; skipping curve plot.")

    print(f"Best val Dice: {best_dice:.4f}. Checkpoint saved to {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
