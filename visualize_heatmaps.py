import argparse
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from datasets.event_frame_heatmap import EventFrameHeatmapDataset


def load_sequence_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_indices(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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
    parser.add_argument("--labels-root", type=str, default="event_data/test")
    parser.add_argument("--list-file", type=str,
                        default="dataset/test_files.txt")
    parser.add_argument("--seqs", type=str, default="3_1")
    parser.add_argument("--length-events", type=int, default=4000)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--out-dir", type=str, default="heatmap_viz")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--indices", type=str, default="")
    args = parser.parse_args()

    seq_override = parse_sequence_list(args.seqs)
    sequences = seq_override if seq_override is not None else load_sequence_list(
        Path(args.list_file))
    dataset = EventFrameHeatmapDataset(
        sequences=sequences,
        frames_root=Path(args.frames_root),
        labels_root=Path(args.labels_root),
        length_events=args.length_events,
        sigma=args.sigma,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    indices = parse_indices(args.indices)
    if indices is None:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=min(
            args.num_samples, len(dataset)), replace=False).tolist()

    for i, idx in enumerate(indices):
        img_tensor, heatmap_tensor, _valid, label_xy = dataset[idx]
        img = img_tensor.squeeze(0).numpy()
        heatmap = heatmap_tensor.squeeze(0).numpy()
        x, y = label_xy.numpy().tolist()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Event frame")
        axes[0].axis("off")

        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(heatmap, cmap="magma", alpha=0.6)
        axes[1].scatter([x], [y], c="cyan", s=25, marker="x")
        axes[1].set_title("Heatmap overlay")
        axes[1].axis("off")

        out_path = Path(args.out_dir) / f"heatmap_{i:03d}_idx_{idx}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
