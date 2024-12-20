"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.
"""
__all__ = [
    "Conv"
]

import torch
from torch import nn


class Conv(nn.Module):
    """
    Dynamically constructs an n-dimensional convolutional layer (Conv1d, Conv2d, or Conv3d) based on
    the input dimensionality `ndim`.

    This module enables flexible construction of convolutional layers for 1D, 2D, or 3D data, by
    internally selecting the appropriate PyTorch convolution operation (`torch.nn.Conv1d`,
    `torch.nn.Conv2d`, or `torch.nn.Conv3d`).
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize the `Conv` module.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).

            - 1: Uses `torch.nn.Conv1d` and expects input tensors of shape
              `(N, C, L)`, where `N` is the batch size, `C` is the number of
              input channels, and `L` is the length of the input sequence.

            - 2: Uses `torch.nn.Conv2d` and expects input tensors of shape
              `(N, C, H, W)`, where `H` and `W` are the spatial dimensions of the input image or
              feature map.

            - 3: Uses `torch.nn.Conv3d` and expects input tensors of shape
              `(N, C, D, H, W)`, where `D`, `H`, and `W` are the spatial dimensions of the input
              image or feature map.

        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple
            Size of the convolving kernel.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels.
            Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        """
        super(Conv, self).__init__()
        # Need the base name of the conv to append the ending (making it a complete attribute)
        base_attr_name = 'Conv'

        # Init the dictionary of different spatial dimensions
        conv_attr_suffix_dict = {1: '1d', 2: '2d', 3: '3d'}

        # Identify the appropriate suffix based on the specified `ndim`
        if ndim not in conv_attr_suffix_dict:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Complete the name of the convolution attribute so we can now import
        conv_attr_name = base_attr_name + conv_attr_suffix_dict[ndim]

        # Import the convolution class with appropriate number of spatial dimensions
        self.conv = getattr(nn, conv_attr_name)
        # Init the conv class with necessary input arguments
        self.conv = self.conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Convolved output tensor.
        """
        return self.conv(input_tensor)