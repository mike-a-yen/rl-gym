import torch
import torch.nn as nn


def compute_output_size(input_shape, layer):
    h_in, w_in, *_ = input_shape
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    kernel = layer.kernel_size
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
    return int(h_out), int(w_out)


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
        self.w_in, self.h_in, self.c_in = self.input_shape

        self.hidden_size = hidden_size
        self.conv_channels = 64
        self.kernel_size = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.c_in, self.conv_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # (B, C, h_in - 2, w_in - 2)
        h_out, w_out = compute_output_size(
            compute_output_size(
                (self.h_in, self.w_in),
                self.conv1[0]
            ),
            self.conv1[2]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        h_out, w_out = compute_output_size(
            compute_output_size(
                (h_out, w_out),
                self.conv2[0]
            ),
            self.conv2[2]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        h_out, w_out = compute_output_size(
            compute_output_size(
                (h_out, w_out),
                self.conv3[0]
            ),
            self.conv3[2]
        )
        self.flat_size = (h_out * w_out) * self.conv_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, X):
        #X_t = X.transpose(1, 3).transpose(2, 3)  # (B, C, W, H)
        c1 = self.conv1(X)
        c2 = self.conv2(c1)
        c_out = self.conv3(c2)
        cout_flat = c_out.reshape(-1, self.flat_size)
        return self.mlp(cout_flat)
