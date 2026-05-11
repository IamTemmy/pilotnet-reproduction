"""
Training script for PilotNet.

Trains the network on augmented Udacity simulator data, validates on a
held-out split, and saves checkpoints when validation loss improves.

Usage:
    python -m src.train --data-dir data/udacity --epochs 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.augment import AugmentedUdacityDataset
from src.model import PilotNet


# Training hyperparameters. Defaults align with common PilotNet reproductions.
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 30
DEFAULT_VAL_FRACTION = 0.20
RANDOM_SEED = 42  # for reproducible train/val splits


def pick_device() -> torch.device:
    """Use Apple Silicon's GPU when available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_train_val_datasets(
    data_dir: str,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[Subset, Subset]:
    """
    Build train and validation datasets without data leakage.

    We split on the RAW CSV rows first, then expose all 6 augmented variants
    of each row only inside its assigned split. This guarantees no raw frame
    appears in both training and validation, even via augmentation.
    """
    full_dataset = AugmentedUdacityDataset(data_dir)
    n_rows = full_dataset.n_rows
    n_variants = full_dataset.n_variants

    # Shuffle raw row indices reproducibly.
    rng = np.random.default_rng(seed)
    row_indices = np.arange(n_rows)
    rng.shuffle(row_indices)

    n_val_rows = int(n_rows * val_fraction)
    val_rows = set(row_indices[:n_val_rows].tolist())
    train_rows = set(row_indices[n_val_rows:].tolist())

    # Build flat sample indices: each raw row contributes n_variants augmented samples.
    train_indices = [r * n_variants + v for r in train_rows for v in range(n_variants)]
    val_indices   = [r * n_variants + v for r in val_rows   for v in range(n_variants)]

    return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Run one full pass over the training data, updating the model's weights."""
    model.train()  # tells PyTorch we're in training mode (matters for some layer types)
    total_loss = 0.0
    n_samples = 0

    for images, targets in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # shape (batch, 1) to match model output

        # Forward pass: predict steering angles for this batch.
        predictions = model(images)
        loss = loss_fn(predictions, targets)

        # Backward pass: compute gradients and update weights.
        optimizer.zero_grad()   # reset gradients from previous step
        loss.backward()         # compute new gradients
        optimizer.step()        # apply them to the weights

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / n_samples


@torch.no_grad()  # disable gradient tracking -- we're not training, just measuring
def evaluate(model, loader, loss_fn, device):
    """Run one full pass over the validation data, measuring loss without updating weights."""
    model.eval()  # tells PyTorch we're in evaluation mode
    total_loss = 0.0
    n_samples = 0

    for images, targets in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        predictions = model(images)
        loss = loss_fn(predictions, targets)

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / n_samples    


def main():
    parser = argparse.ArgumentParser(description="Train PilotNet on Udacity simulator data.")
    parser.add_argument("--data-dir", type=str, default="data/udacity",
                        help="Path to the Udacity data folder (containing driving_log.csv and IMG/).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes. Use 0 if you hit weird crashes.")
    args = parser.parse_args()

    # Set up reproducibility.
    torch.manual_seed(RANDOM_SEED)

    device = pick_device()
    print(f"Training device: {device}")

    # Build datasets and loaders.
    train_dataset, val_dataset = make_train_val_datasets(args.data_dir, args.val_fraction)
    print(f"Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # Build the model, loss function, and optimizer.
    model = PilotNet().to(device)
    loss_fn = torch.nn.MSELoss()  # mean squared error -- standard for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare checkpoint and history tracking.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    # The main training loop.
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"  train loss: {train_loss:.5f}   val loss: {val_loss:.5f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save checkpoint if this is our best validation performance so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pth")
            print(f"  ✓ saved new best model (val loss {val_loss:.5f})")

    # Save the final model and the training history.
    torch.save(model.state_dict(), checkpoint_dir / "final.pth")
    pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
    print(f"\nDone. Best val loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()

