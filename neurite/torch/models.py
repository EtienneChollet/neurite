"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""
__all__ = [
    "BasicUNet"
]

from typing import List, Union
import torch
from torch import nn
from . import modules, utils


class BasicUNet(nn.Module):
    """
    Flexible unet with many configuration options!

    Attributes
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.

    Notes
    -----
    - This UNet is based on the architecture of the UNet found in the paper by Olaf Ronneberger:
    https://arxiv.org/pdf/1505.04597

    Examples
    --------
    >>> model = BasicUNet(
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     norms='instance', activations=nn.ReLU
    ... )
    >>> input_tensor = torch.randn(1, 1, 128, 128)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 128, 128])
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: List[int] = (16, 16, 16, 16, 16),
        norms: Union[List[str], str] = None,
        activations: Union[List[str], str] = nn.ReLU,
        order: str = 'ncaca',
        final_activation: Union[str, nn.Module, None] = nn.Sigmoid(),
    ):
        """
        Instantiate `BasicUNet`

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        norms : Union[List[str], str, None], optional
            Normalization layers to use in each block. Can be a string or a list
            of strings specifying norms for each layer, or `None` for no norm.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        order : str, optional
            Order of operations in each convolutional block (e.g., 'ncaca').

        Raises
        ------

        """
        super().__init__()
        # Normalization layers
        if not isinstance(norms, list):
            self.norms = [norms] * len(nb_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(nb_features)

        # Original sequence for downsampling conv blocks
        self.nb_features = [in_channels, *nb_features]

        # Inverted sequence for upsampling conv blocks
        self.reversed_features = list(reversed(nb_features))

        # Downsampling convolutional blocks
        self.downsampling_conv_blocks = utils.make_downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features,
            norms=self.norms,
            activations=self.activations,
            order=order
        )

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        self.lowest_resolution_conv_block = modules.ConvBlock(
            ndim=ndim,
            in_channels=self.nb_features[-1],
            out_channels=self.nb_features[-1],
            order=order,
        )

        # Upsampling convolutional blocks
        self.upsampling_conv_blocks = utils.make_upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.reversed_features,
            norms=self.norms,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
        )

        # Final convolutional block
        self.out_layer = modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
        )

    def forward(self, feature_tensor: torch.Tensor):
        """
        Forward pass through the `BasicUNet` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """
        # Downsampling path
        skip_connections = []
        for downsampling_conv_block in self.downsampling_conv_blocks:
            feature_tensor, residual = downsampling_conv_block(feature_tensor, return_residual=True)
            skip_connections.append(residual)  # Save for skip connection

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # Upsampling path
        for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
            skip = skip_connections[-(i + 1)]
            feature_tensor = upsampling_conv_block(feature_tensor, skip)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor
