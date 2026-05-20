from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_label_line(line: str) -> Tuple[int, int, int]:
    parts = line.strip().strip("()").split(",")
    if len(parts) < 2:
        raise ValueError(f"Invalid label line: {line}")
    x = int(float(parts[0]))
    y = int(float(parts[1]))
    blink = int(float(parts[2])) if len(parts) > 2 else 0
    return x, y, blink


def get_label_filename(length_events: int) -> str:
    if length_events == 2000:
        return "frame_label_interpolated2000.txt"
    if length_events == 4000:
        return "frame_label_interpolated4.txt"
    if length_events == 5000:
        return "frame_label_interpolated5000.txt"
    if length_events == 6000:
        return "frame_label_interpolated6000.txt"
    if length_events == 8000:
        return "frame_label_interpolated8000.txt"
    raise ValueError(f"Unsupported length_events: {length_events}")


class EventFrameHeatmapDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        frames_root: Path,
        labels_root: Path,
        length_events: int,
        sigma: float,
        translate_px: int = 0,
        translate_prob: float = 0.0,
    ) -> None:
        self.samples: List[Tuple[Path, Tuple[int, int, int]]] = []
        self.frames_root = frames_root
        self.labels_root = labels_root
        self.length_events = length_events
        self.sigma = float(sigma)
        self.translate_px = int(translate_px)
        self.translate_prob = float(translate_prob)
        self._rng = np.random.default_rng()
        self._grid = None
        self._grid_shape = None

        label_filename = get_label_filename(length_events)
        missing = []

        for seq in sequences:
            seq_frames = frames_root / seq
            seq_labels = labels_root / seq / label_filename

            if not seq_frames.is_dir() or not seq_labels.is_file():
                missing.append(seq)
                continue

            images = sorted(
                [p for p in seq_frames.iterdir() if p.suffix.lower()
                 in [".jpg", ".png"]]
            )
            if not images:
                missing.append(seq)
                continue

            with open(seq_labels, "r", encoding="utf-8") as f:
                labels = [parse_label_line(line) for line in f if line.strip()]

            count = min(len(images), len(labels))
            for i in range(count):
                self.samples.append((images[i], labels[i]))

        if missing:
            print(
                f"Warning: {len(missing)} sequences missing frames or labels. Example: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_grid(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._grid is None or self._grid_shape != (height, width):
            ys = torch.arange(height, dtype=torch.float32).view(height, 1)
            xs = torch.arange(width, dtype=torch.float32).view(1, width)
            self._grid = (ys, xs)
            self._grid_shape = (height, width)
        return self._grid

    def _translate_image(self, img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return img

        _, height, width = img.shape
        out = torch.zeros_like(img)

        src_y_start = max(0, -dy)
        src_y_end = min(height, height - dy)
        dst_y_start = max(0, dy)
        dst_y_end = min(height, height + dy)

        src_x_start = max(0, -dx)
        src_x_end = min(width, width - dx)
        dst_x_start = max(0, dx)
        dst_x_end = min(width, width + dx)

        if src_y_start >= src_y_end or src_x_start >= src_x_end:
            return out

        out[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[:,
                                                                   src_y_start:src_y_end, src_x_start:src_x_end]
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, (x, y, _blink) = self.samples[idx]

        img = Image.open(img_path).convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        if self.translate_px > 0 and self._rng.random() < self.translate_prob:
            dx = int(self._rng.integers(-self.translate_px, self.translate_px + 1))
            dy = int(self._rng.integers(-self.translate_px, self.translate_px + 1))
            img_tensor = self._translate_image(img_tensor, dx, dy)
            x += dx
            y += dy

        height, width = img_tensor.shape[-2], img_tensor.shape[-1]
        ys, xs = self._get_grid(height, width)

        if x < 0 or y < 0 or x >= width or y >= height:
            heatmap = torch.zeros((height, width), dtype=torch.float32)
        else:
            dx = xs - float(x)
            dy = ys - float(y)
            heatmap = torch.exp(-(dx * dx + dy * dy) /
                                (2.0 * self.sigma * self.sigma))

        heatmap = heatmap.unsqueeze(0)
        label_xy = torch.tensor([float(x), float(y)], dtype=torch.float32)
        valid = torch.tensor(1.0, dtype=torch.float32)

        return img_tensor, heatmap, valid, label_xy
