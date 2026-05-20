import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from datasets.event_frame_heatmap import EventFrameHeatmapDataset
from models.unet_small import UNetSmall
from trainers.heatmap_trainer import evaluate, train_one_epoch


def load_sequence_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_sequence_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-root",
        type=str,
        default="vis_result_for_frst/imgs_lengthsplit_4000_amp",
    )
    parser.add_argument("--labels-root", type=str, default="event_data/train")
    parser.add_argument("--train-list", type=str,
                        default="dataset/train_files.txt")
    parser.add_argument("--val-list", type=str,
                        default="dataset/val_files.txt")
    parser.add_argument("--train-seqs", type=str, default="")
    parser.add_argument("--val-seqs", type=str, default="")
    parser.add_argument("--length-events", type=int, default=4000)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--translate-px", type=int, default=0)
    parser.add_argument("--translate-prob", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-path", type=str,
                        default="ckpt/unet_heatmap.pth")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    labels_root = Path(args.labels_root)
    train_list = Path(args.train_list)
    val_list = Path(args.val_list)

    train_override = parse_sequence_list(args.train_seqs)
    val_override = parse_sequence_list(args.val_seqs)

    train_seqs = train_override if train_override is not None else load_sequence_list(
        train_list)
    val_seqs = val_override if val_override is not None else load_sequence_list(
        val_list)

    train_ds = EventFrameHeatmapDataset(
        train_seqs,
        frames_root,
        labels_root,
        args.length_events,
        args.sigma,
        translate_px=args.translate_px,
        translate_prob=args.translate_prob,
    )
    val_ds = EventFrameHeatmapDataset(
        val_seqs,
        frames_root,
        labels_root,
        args.length_events,
        args.sigma,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = UNetSmall(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_err = float("inf")
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_err = train_one_epoch(
            model, train_loader, optimizer, device)
        val_loss, val_err = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.6f} train_err={train_err:.2f} "
            f"val_loss={val_loss:.6f} val_err={val_err:.2f}"
        )

        if val_err < best_err:
            best_err = val_err
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_err": val_err},
                args.save_path,
            )
            print(
                f"Saved best model to {args.save_path} (val_err={val_err:.2f})")


if __name__ == "__main__":
    main()
