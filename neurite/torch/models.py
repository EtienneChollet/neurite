"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""
__all__ = [
    "FlexibleUNet"
]

import torch
from torch import nn


class FlexibleUNet(nn.Module):
    """
    Flexible unet with many configuration options!

    Attributes
    ----------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self):
        """
        Instantiate `FlexibleUNet`

        Parameters
        ----------

        Raises
        ------

        """
        super().__init__()
        raise NotImplementedError(
            "FlexibleUNet is not yet ready :("
        )

    def forward(self, input_tensor: torch.Tensor):
        """
        Forward pass through the FlexibleUNet model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """
        raise NotImplementedError(
            "forward() method of FlexibleUNet"
        )
