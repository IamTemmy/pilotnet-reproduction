"""
Data augmentation for the Udacity PilotNet pipeline.

Two augmentations, both from the original paper:

1. Multi-camera augmentation: treat the left- and right-camera images as if they
   were center-camera images taken from a car that had drifted off-center.
   Pair them with an adjusted steering label that would correct that drift.
   This triples the effective dataset size.

Combined, ~8,000 raw frames become ~48,000 effective training examples.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import preprocess


# Community-converged value for Udacity-simulator PilotNet reproductions.
# Range -1.0 to 1.0 maps to full-left to full-right steering.
STEERING_CORRECTION = 0.20


class AugmentedUdacityDataset(Dataset):
    """
    Augmented version of the Udacity driving dataset.

    Each row in driving_log.csv yields 6 training examples:
        (center image, steering)
        (left image,   steering + correction)
        (right image,  steering - correction)
        (flipped center, -steering)
        (flipped left,   -(steering + correction))
        (flipped right,  -(steering - correction))

    So 8,036 raw rows yield ~48,216 augmented samples.
    """

    # The 6 (camera, flip) combinations we generate per CSV row.
    # Each tuple is (camera_column, steering_adjustment, do_flip).
    VARIANTS = [
        ("center", 0.0,                  False),
        ("left",   +STEERING_CORRECTION, False),
        ("right",  -STEERING_CORRECTION, False),
        ("center", 0.0,                  True),
        ("left",   +STEERING_CORRECTION, True),
        ("right",  -STEERING_CORRECTION, True),
    ]

    def __init__(self, data_dir: str | Path):
        import pandas as pd

        self.data_dir = Path(data_dir)
        csv_path = self.data_dir / "driving_log.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected driving_log.csv at {csv_path}")

        column_names = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
        first_row = pd.read_csv(csv_path, header=None, nrows=1)
        try:
            float(first_row.iloc[0, 3])
            self.log = pd.read_csv(csv_path, header=None, names=column_names)
        except ValueError:
            self.log = pd.read_csv(csv_path, header=0, names=column_names)

        for col in ("center", "left", "right"):
            self.log[col] = self.log[col].str.strip()

        self.n_rows = len(self.log)
        self.n_variants = len(self.VARIANTS)

    def __len__(self) -> int:
        return self.n_rows * self.n_variants

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Decode the flat index into (csv row, variant) -- which raw frame, which augmentation.
        row_idx = index // self.n_variants
        variant_idx = index % self.n_variants
        camera, steering_adjustment, do_flip = self.VARIANTS[variant_idx]

        row = self.log.iloc[row_idx]
        image_path = self.data_dir / "IMG" / Path(row[camera]).name

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Flip BEFORE preprocessing so the crop/resize apply to the flipped frame.
        if do_flip:
            bgr = cv2.flip(bgr, 1)  # 1 = horizontal flip

        image = preprocess(bgr)
        steering = float(row["steering"]) + steering_adjustment
        if do_flip:
            steering = -steering

        return torch.from_numpy(image), torch.tensor(steering, dtype=torch.float32)


if __name__ == "__main__":
    dataset = AugmentedUdacityDataset("data/udacity")
    print(f"Augmented dataset size: {len(dataset)} samples")
    print(f"(Raw rows: {dataset.n_rows}, variants per row: {dataset.n_variants})")

    # Inspect a few samples to confirm the augmentation works.
    for idx in [0, 1, 2, 3, 4, 5]:
        image, steering = dataset[idx]
        camera, adj, flip = AugmentedUdacityDataset.VARIANTS[idx]
        print(f"  Sample {idx}: camera={camera:6s} flip={str(flip):5s} steering={steering.item():+.4f}")