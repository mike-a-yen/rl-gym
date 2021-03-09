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


def make_conv(input_shape, num_kernels, kernel_size, stride=1):
    """Stack a conv/activation layer and compute the output size.

    Parameters
    ----------
    input_shape : tuple
        expected input shape (w, h, c)
    num_kernels : int
    kernel_size : int
    stride : int, optional
        kernel stride, by default 1

    Returns
    -------
    conv_layer, output_shape (w, h, c)
    """
    w_in, h_in, c_in = input_shape
    conv = nn.Sequential(
        nn.Conv2d(c_in, num_kernels, kernel_size, stride=stride),
        nn.ReLU()
    )
    h_out, w_out = compute_output_size((h_in, w_in), conv[0])
    return conv, (w_out, h_out, num_kernels)


class GymConvModel(nn.Module):
    def __init__(self, input_shape, convs, hidden_size, output_size):
        super().__init__()
        self.input_shape = input_shape  # (H, W, C)
        self.output_size = output_size
        self.w_in, self.h_in, self.c_in = self.input_shape
        self.hidden_size = hidden_size

        conv_layers, in_shape = [], (self.w_in, self.h_in, self.c_in)
        for conv_config in convs:
            layer, out_shape = make_conv(in_shape, conv_config.num_kernels, conv_config.kernel_size, conv_config.stride)
            conv_layers.append(layer)
            in_shape = out_shape

        self.conv = nn.Sequential(*conv_layers)
        self.flat_size = out_shape[0] * out_shape[1] * out_shape[2]
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, X):
        c_out = self.conv(X)
        cout_flat = c_out.reshape(-1, self.flat_size)
        return self.mlp(cout_flat)
