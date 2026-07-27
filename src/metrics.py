
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(pred_probs, target, eps=1e-7):
    """pred_probs, target: (B,1,H,W) in [0,1]. Returns mean Dice over batch."""
    pred_flat = pred_probs.reshape(pred_probs.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred_probs > threshold).float()
    pred_flat = pred_bin.reshape(pred_bin.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def precision_recall(pred_probs, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred_probs > threshold).float()
    pred_flat = pred_bin.reshape(pred_bin.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.mean(), recall.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss. Dice handles class imbalance (nuclei are
    a small fraction of pixels); BCE gives stable per-pixel gradients.
    """

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice_loss = 1 - dice_coefficient(probs, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_dice, total_iou, total_prec, total_rec, n = 0.0, 0.0, 0.0, 0.0, 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)

        d = dice_coefficient(probs, masks).item()
        i = iou_score(probs, masks).item()
        p, r = precision_recall(probs, masks)

        bs = imgs.size(0)
        total_dice += d * bs
        total_iou += i * bs
        total_prec += p.item() * bs
        total_rec += r.item() * bs
        n += bs

    return {
        "dice": total_dice / n,
        "iou": total_iou / n,
        "precision": total_prec / n,
        "recall": total_rec / n,
    }
