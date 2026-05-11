"""
Dataset loader for the Udacity Self-Driving Car Simulator data.

Reads the driving_log.csv produced by the simulator, loads the corresponding
center-camera images from disk, applies the PilotNet preprocessing pipeline
(crop -> resize -> RGB), and returns (image_tensor, steering_angle) pairs.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# PilotNet input shape, per Section 4 of the paper.
INPUT_HEIGHT = 66
INPUT_WIDTH = 200

# Crop boundaries for the 160x320 simulator frames.
# We drop the top portion (sky, trees) and the bottom portion (car hood).
# These values are standard for Udacity-simulator-based PilotNet implementations.
CROP_TOP = 60
CROP_BOTTOM = 135  # everything below this is the car hood


def preprocess(bgr_image: np.ndarray) -> np.ndarray:
    """
    Apply the PilotNet input pipeline to a single raw simulator frame.

    Args:
        bgr_image: H x W x 3 image as loaded by OpenCV (BGR channel order).

    Returns:
        Float32 array of shape (3, 66, 200), pixel range [0, 255], RGB order.
        This is the format the model's forward() expects.
    """
    # 1. Crop out the sky and hood. Keep only the road band.
    cropped = bgr_image[CROP_TOP:CROP_BOTTOM, :, :]

    # 2. Resize the road band to the network's expected input size.
    #    cv2.resize wants (width, height) -- order matters and trips people up.
    resized = cv2.resize(cropped, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)

    # 3. Convert BGR (OpenCV's default) to RGB (what the paper uses,
    #    and what humans/torchvision expect).
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 4. Transpose from (H, W, C) to (C, H, W) -- PyTorch's convention.
    chw = rgb.transpose(2, 0, 1)

    # 5. Return as float32. Pixel values stay in [0, 255]; the model
    #    normalizes to [-1, 1] internally.
    return chw.astype(np.float32)


class UdacityDrivingDataset(Dataset):
    """
    PyTorch Dataset over the Udacity simulator's driving_log.csv.

    Each item is a (image_tensor, steering_angle_tensor) pair drawn from the
    center camera. Left/right cameras and augmentation are handled separately
    in augment.py.
    """

    def __init__(self, data_dir: str | Path):
        """
        Args:
            data_dir: Path to the udacity data folder (the one containing
                      driving_log.csv and IMG/).
        """
        self.data_dir = Path(data_dir)
        csv_path = self.data_dir / "driving_log.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Expected driving_log.csv at {csv_path}")

        # The Udacity sample CSV has a header row; some simulator-generated
        # CSVs do not. We detect this by checking whether the first row
        # parses as a number in the steering column.
        column_names = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
        first_row = pd.read_csv(csv_path, header=None, nrows=1)
        try:
            float(first_row.iloc[0, 3])
            self.log = pd.read_csv(csv_path, header=None, names=column_names)
        except ValueError:
            self.log = pd.read_csv(csv_path, header=0, names=column_names)

        # Strip any whitespace from the image path columns -- common CSV gotcha.
        for col in ("center", "left", "right"):
            self.log[col] = self.log[col].str.strip()

    def __len__(self) -> int:
        return len(self.log)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.log.iloc[index]

        # The CSV stores paths like "IMG/center_2016_..." -- relative to data_dir.
        # Some versions store absolute paths; handle both by always taking the
        # filename and rejoining against our local IMG/ folder.
        image_path = self.data_dir / "IMG" / Path(row["center"]).name

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = preprocess(bgr)
        steering = float(row["steering"])

        return torch.from_numpy(image), torch.tensor(steering, dtype=torch.float32)


if __name__ == "__main__":
    # Smoke test: load the Udacity data and sanity-check the first sample.
    dataset = UdacityDrivingDataset("data/udacity")
    print(f"Dataset size: {len(dataset)} samples")

    image, steering = dataset[0]
    print(f"Image tensor shape: {tuple(image.shape)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image pixel range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"Steering value: {steering.item():.4f}")
    print(f"Steering dtype: {steering.dtype}")