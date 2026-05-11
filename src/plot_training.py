"""
Plot training curves from the training_history.csv saved by train.py.

Usage:
    python -m src.plot_training
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-csv", type=str, default="checkpoints/training_history.csv")
    parser.add_argument("--output", type=str, default="results/training_curves.png")
    args = parser.parse_args()

    df = pd.read_csv(args.history_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], label="Training loss", linewidth=2, marker="o", markersize=4)
    ax.plot(df["epoch"], df["val_loss"], label="Validation loss", linewidth=2, marker="s", markersize=4)

    # Mark the best epoch.
    best_idx = df["val_loss"].idxmin()
    best_epoch = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_loss"]
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best epoch ({best_epoch}, val loss {best_val:.5f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("PilotNet Training Curves — Udacity Simulator Data")
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved training curves to {output_path}")


if __name__ == "__main__":
    main()