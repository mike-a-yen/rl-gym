import torch
import torch.nn as nn


class QGymModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, X):
        return self.layers(X)


class GymConvModel(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super().__init__()
        self.input_shape = input_shape  # (H, W, C)
        self.output_size = output_size
        self.h_in, self.w_in, self.c_in = self.input_shape

        self.hidden_size = hidden_size
        self.conv_channels = 64
        self.kernel_size = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.c_in, self.conv_channels, kernel_size=self.kernel_size),
            nn.ReLU()
        )  # (B, C, H, W)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=self.kernel_size),
            nn.ReLU()
        )
        self.flat_size = (self.h_in - 4) * (self.w_in - 4) * self.conv_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, X):
        X_t = X.transpose(1, 3).transpose(2, 3)  # (B, C, W, H)
        c1 = self.conv1(X_t)
        c2 = self.conv2(c1)
        c2_flat = c2.reshape(-1, self.flat_size)
        return self.mlp(c2_flat)
