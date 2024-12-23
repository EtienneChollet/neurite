"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.
"""
__all__ = [
    "Conv",
    "ConvBlock"
]

from typing import Union
import torch
from torch import nn
from .layers import Norm, Activation


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

        # Mapping of spatial dimensions for convolutions
        conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

        # Determine if `ndim` is valid
        if ndim not in conv_dim_map:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Dynamically retreive nn.convXd
        conv_cls_name = f"Conv{conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Init the dynamically retreived conv class with necessary input arguments
        self.conv = conv_cls(
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


class ConvBlock(nn.Sequential):
    """
    Convolutional Block comprising a convolutional layer, and optionally, an activation function and
    normalization.

    The default sequence of operations in this block is:
        1. **Convolution**: Applies a 2D convolution over the input.
        2. **Normalization**: Normalizes the output of the convolution to stabilize and accelerate
        training.
        3. **Activation Function**: Introduces non-linearity to the model.

    Attributes
    ----------
    conv : nn.Conv2d
        The convolutional layer.
    batch_norm : nn.BatchNorm2d
        The batch normalization layer.
    activation : nn.Module
        The activation function.

    Examples
    --------
    >>> import torch.nn as nn
    >>> conv_block = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU()
        )
    >>> input_tensor = torch.randn(16, 64, 32, 32)
    >>> output = conv_block(input_tensor)
    >>> print(output.shape)
    torch.Size([16, 128, 32, 32])
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
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = None,
        order: str = 'nca'
    ):
        """
        Initialize the `ConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple, optional
            Size of the convolving kernel. Default is 3.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels. Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        norm : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Norm` module: Instantiated or uninstantiated `Norm` layer.
                e.g. nn.InstanceNorm3d(16) or nn.InstanceNorm3d
            - `None`: No normalization is applied. Default is `None`.

        activation : str, nn.Module, or None, optional
            Defines the activation layer. Can be one of:
            - A string: Supported options are 'relu', 'leaky_relu', or 'elu'.
            - A `nn.Module`: Instantiated or uninstantiated activation module.
                e.g. nn.Sigmoid(), nn.Sigmoid
            - `None`: No activation is applied. Default is `None`.

        order : str, optional
            The order of operations in the block. Default is 'nca'
            (normalization -> convolution -> activation).
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation

        Examples
        --------
        ### Basic usage with default options:
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm="batch",
                activation="relu"
            )
        >>> input_tensor = torch.randn(1, 16, 64, 64)
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Using a pre-instantiated `Norm` and `Activation` module:
        >>> norm_layer = nn.BatchNorm2d(32)
        >>> activation_layer = nn.ReLU()
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm=norm_layer,
                activation=activation_layer
            )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Omitting normalization or activation:
        >>> conv_block = ConvBlock(
            ndim=2,
            in_channels=16,
            out_channels=32,
            norm=None,
            activation=None
        )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])
        """
        self.order = order
        if order is None:
            order = 'nca'
        # Break order into list of letters (operations)
        self.order = list(order)

        # Validate the operations
        valid_operations = ['c', 'n', 'a']
        if not set(order).issubset(valid_operations):
            raise ValueError(
                f"Invalid order. Must be a subset of {valid_operations}."
            )

        # Init layers container
        layers = []
        # Collect layers in the appropriate order
        for operation in self.order:
            if operation == 'c':
                # Make convolution
                layers.append(
                    Conv(
                        ndim,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias
                    )
                )

            elif operation == 'n' and norm is not None:
                # Make normalization
                layers.append(Norm(norm, ndim, out_channels))

            elif operation == 'a' and activation is not None:
                # Make activation
                layers.append(Activation(activation))

        # Create the Sequential container
        super(ConvBlock, self).__init__(*layers)
