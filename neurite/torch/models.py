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
    encoders : nn.ModuleList
        Encoder layers.
    bottleneck : nn.Module
        Central bottleneck layer.
    decoders : nn.ModuleList
        Decoder layers.
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
            Number of features at each layer of the encoder. Must be a list of
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

        # Original sequence for encoder
        self.nb_features = [in_channels, *nb_features]
        # Inverted sequence for decoder
        self.reversed_features = list(reversed(nb_features))

        # Encoder(s)
        self.encoders = utils.make_encoders(
            ndim=ndim,
            nb_features=self.nb_features,
            norms=self.norms,
            activations=self.activations,
            order=order
        )

        # Bottleneck
        self.bottleneck = modules.ConvBlock(
            ndim=ndim,
            in_channels=self.nb_features[-1],
            out_channels=self.nb_features[-1],
            order=order,
        )

        # Decoders
        self.decoders = utils.make_decoders(
            ndim=ndim,
            nb_features=self.reversed_features,
            norms=self.norms,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
        )

        # Final convolution
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
        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            feature_tensor, residual = encoder(feature_tensor, return_residual=True)
            skip_connections.append(residual)  # Save for skip connection

        # Bottleneck
        feature_tensor = self.bottleneck(feature_tensor)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            feature_tensor = decoder(feature_tensor, skip)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor
