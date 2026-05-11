"""
PilotNet — a faithful PyTorch reproduction of the architecture from:
    Bojarski et al., "End to End Learning for Self-Driving Cars", NVIDIA, 2016.
    https://arxiv.org/abs/1604.07316

Architecture (Figure 4 of the paper):
    Input: 3-channel RGB image, 66x200
    -> Normalization (hardcoded, non-learnable)
    -> 5 convolutional layers (3 with 5x5 kernels stride 2, 2 with 3x3 kernels stride 1)
    -> Flatten
    -> 3 fully connected hidden layers (100, 50, 10)
    -> Output: 1 scalar (steering angle in radians)
    Total parameters: ~250k
"""

import torch
import torch.nn as nn


class PilotNet(nn.Module):
    """End-to-end CNN that maps a forward-facing camera image to a steering angle."""

    def __init__(self):
        super().__init__()

        # 5 convolutional layers. The paper uses ELU (Exponential Linear Unit) as the
        # activation function. We omit padding to match the output sizes given in the paper.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        # After the conv stack, the feature map is 64 channels x 1 height x 18 width
        # = 1152 features. We flatten and feed into the fully connected head.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 1 * 18, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1),  # final layer: no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, 3, 66, 200) with pixel values in [0, 255].
        Returns:
            tensor of shape (batch, 1) — predicted steering angle in radians.
        """
        # Hardcoded normalization, as specified in the paper.
        # Maps pixel range [0, 255] -> [-1, 1] without learnable parameters.
        x = x / 127.5 - 1.0

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    # Sanity check: create a model and run a fake batch through it.
    # This is a smoke test you can run with: python src/model.py
    model = PilotNet()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PilotNet — {n_params:,} trainable parameters")

    fake_batch = torch.randn(4, 3, 66, 200) * 127.5 + 127.5  # fake [0, 255] images
    output = model(fake_batch)
    print(f"Input shape:  {tuple(fake_batch.shape)}")
    print(f"Output shape: {tuple(output.shape)}")