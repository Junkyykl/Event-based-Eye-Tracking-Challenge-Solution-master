from typing import Tuple

import torch
from tqdm import tqdm


def compute_centers_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    b, _, h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    idx = torch.argmax(flat, dim=1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    return torch.stack([xs, ys], dim=1)


def train_one_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    loss_total = 0.0
    err_total = 0.0
    valid_total = 0.0

    for imgs, targets, valids, label_xy in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        valids = valids.to(device)
        label_xy = label_xy.to(device)

        logits = model(imgs)
        preds = torch.sigmoid(logits)

        loss_map = (preds - targets).pow(2).view(preds.size(0), -1).mean(dim=1)
        loss = (loss_map * valids).sum() / torch.clamp(valids.sum(), min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_xy = compute_centers_from_heatmap(preds)
            err = torch.norm(pred_xy - label_xy, dim=1)
            err = (err * valids).sum()

        loss_total += loss.item() * imgs.size(0)
        err_total += err.item()
        valid_total += valids.sum().item()

    avg_loss = loss_total / max(len(loader.dataset), 1)
    avg_err = err_total / max(valid_total, 1.0)
    return avg_loss, avg_err


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    loss_total = 0.0
    err_total = 0.0
    valid_total = 0.0

    with torch.no_grad():
        for imgs, targets, valids, label_xy in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            valids = valids.to(device)
            label_xy = label_xy.to(device)

            logits = model(imgs)
            preds = torch.sigmoid(logits)

            loss_map = (
                preds - targets).pow(2).view(preds.size(0), -1).mean(dim=1)
            loss = (loss_map * valids).sum() / \
                torch.clamp(valids.sum(), min=1.0)

            pred_xy = compute_centers_from_heatmap(preds)
            err = torch.norm(pred_xy - label_xy, dim=1)
            err = (err * valids).sum()

            loss_total += loss.item() * imgs.size(0)
            err_total += err.item()
            valid_total += valids.sum().item()

    avg_loss = loss_total / max(len(loader.dataset), 1)
    avg_err = err_total / max(valid_total, 1.0)
    return avg_loss, avg_err
