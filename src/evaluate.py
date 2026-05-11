"""
Evaluate a trained PilotNet on held-out validation data.

Produces three artifacts in results/:
  - predictions_grid.png   : visual side-by-side of predicted vs actual steering
                             across a representative sample of validation images.
  - evaluation_summary.txt : aggregate statistics (MAE, MSE, error distribution)
                             over the entire validation set.
  - error_histogram.png    : histogram of prediction errors across the val set.

This complements the live simulator demo (results/demo.mp4) with quantitative
evaluation on the full held-out set.

Usage:
    python -m src.evaluate --model checkpoints/best.pth
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import PilotNet
from src.train import make_train_val_datasets, pick_device


def gather_predictions(model, dataset, device, batch_size=64):
    """Run the model over the entire dataset and return (predictions, targets)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            preds = model(images).squeeze(1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def make_predictions_grid(model, dataset, device, output_path, n=24, rows=6, cols=4):
    """
    Pick `n` representative validation samples (sorted by absolute steering, so we
    get a mix of straights and curves), run the model, and render a grid showing
    each image with predicted vs actual steering overlaid.
    """
    preds, targets = gather_predictions(model, dataset, device)
    abs_targets = np.abs(targets)

    sorted_idx = np.argsort(abs_targets)
    step = max(1, len(sorted_idx) // n)
    pick = sorted_idx[::step][:n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    axes = axes.flatten()

    for ax, idx in zip(axes, pick):
        image_tensor, _ = dataset[int(idx)]
        image_display = image_tensor.numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

        pred = preds[idx]
        actual = targets[idx]
        err = pred - actual

        abs_err = abs(err)
        if abs_err < 0.05:
            color = "green"
        elif abs_err < 0.15:
            color = "darkorange"
        else:
            color = "red"

        ax.imshow(image_display)
        ax.set_title(f"Actual: {actual:+.3f}\nPredicted: {pred:+.3f}\nError: {err:+.3f}",
                     fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle("PilotNet Predictions on Held-Out Validation Images",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved predictions grid to {output_path}")

    return preds, targets


def write_evaluation_summary(preds, targets, output_path):
    """Compute aggregate statistics and write a plain-text summary."""
    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = abs_errors.mean()
    mse = (errors ** 2).mean()
    rmse = np.sqrt(mse)
    median_abs_error = np.median(abs_errors)
    p90_abs_error = np.percentile(abs_errors, 90)
    p99_abs_error = np.percentile(abs_errors, 99)
    max_abs_error = abs_errors.max()

    n_within_005 = (abs_errors < 0.05).sum()
    n_within_010 = (abs_errors < 0.10).sum()
    n_within_020 = (abs_errors < 0.20).sum()
    total = len(errors)

    summary = f"""PilotNet — Evaluation on Held-Out Validation Set
==================================================
Validation samples evaluated: {total:,}

Aggregate error metrics (steering range is [-1, 1]):
  Mean Absolute Error (MAE):       {mae:.5f}
  Mean Squared Error (MSE):        {mse:.5f}
  Root Mean Squared Error (RMSE):  {rmse:.5f}
  Median Absolute Error:           {median_abs_error:.5f}
  90th percentile absolute error:  {p90_abs_error:.5f}
  99th percentile absolute error:  {p99_abs_error:.5f}
  Max absolute error:              {max_abs_error:.5f}

Accuracy buckets:
  Predictions within 0.05 of true steering:  {n_within_005:>6,} / {total:,}  ({100 * n_within_005 / total:.1f}%)
  Predictions within 0.10 of true steering:  {n_within_010:>6,} / {total:,}  ({100 * n_within_010 / total:.1f}%)
  Predictions within 0.20 of true steering:  {n_within_020:>6,} / {total:,}  ({100 * n_within_020 / total:.1f}%)

Interpretation:
  The steering signal ranges from -1 (full left) to +1 (full right). An MAE of
  {mae:.3f} means the typical prediction is off by roughly {mae * 100:.1f}% of the
  full steering range -- generally suitable for closed-loop driving on this track.
"""
    Path(output_path).write_text(summary)
    print(f"Saved evaluation summary to {output_path}")
    print(summary)


def make_error_histogram(preds, targets, output_path):
    """Save a histogram of prediction errors -- helps recruiters see the distribution shape."""
    errors = preds - targets

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=60, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Prediction error (predicted - actual steering)")
    ax.set_ylabel("Frequency")
    ax.set_title("PilotNet -- Validation Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    print(f"Saved error histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best.pth")
    parser.add_argument("--data-dir", type=str, default="data/udacity")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Evaluation device: {device}")

    model = PilotNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    _, val_dataset = make_train_val_datasets(args.data_dir)
    print(f"Validation samples: {len(val_dataset):,}")

    preds, targets = make_predictions_grid(
        model, val_dataset, device,
        output_path=output_dir / "predictions_grid.png",
    )
    write_evaluation_summary(preds, targets, output_dir / "evaluation_summary.txt")
    make_error_histogram(preds, targets, output_dir / "error_histogram.png")


if __name__ == "__main__":
    main()