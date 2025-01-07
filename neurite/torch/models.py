"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""
__all__ = [
    "FlexibleUNet"
]

from typing import List, Union
import torch
from torch import nn
from . import modules, utils


# TODO: Decide on the name for this with @adrian.dalca
class FlexibleUNet(nn.Module):
    """
    Flexible unet with many configuration options!

    Attributes
    ----------

    Notes
    -----
    - This UNet is based on the architecture of the UNet found in the paper by Olaf Ronneberger:
    https://arxiv.org/pdf/1505.04597

    Examples
    --------

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
        Instantiate `FlexibleUNet`

        Parameters
        ----------

        Raises
        ------

        """
        super().__init__()
        self.norms = norms
        self.activations = activations
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
            order=order
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
        Forward pass through the FlexibleUNet model.

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
            feature_tensor = encoder(feature_tensor)
            skip_connections.append(feature_tensor)  # Save for skip connection

        # Bottleneck
        feature_tensor = self.bottleneck(feature_tensor)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            # Concat along channel dimension
            feature_tensor = torch.cat((feature_tensor, skip), dim=1)
            feature_tensor = decoder(feature_tensor)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor
