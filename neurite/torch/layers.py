"""
Layers for the neurite project, written in PyTorch.

If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018. https://arxiv.org/abs/1903.03148


Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""
__all__ = [
    "Negate",
    "RescaleValues",
    "Resize",
    "SoftQuantize",
    "MSE"
    ]

import torch
from torch import nn


class Negate(nn.Module):
    """
    A PyTorch module that returns the negative of the input tensor.
    """
    def __init__(self):
        """
        Initialize the `Negate` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Negate` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to negate.
        """
        # Negate the tensor and return it.
        return -input_tensor


class RescaleValues(nn.Module):
    """
    A PyTorch module that rescales the values of the input tensor.
    """
    def __init__(self):
        """
        Initialize the `RescaleValues` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RescaleValues` module.
        """
        raise NotImplementedError("The `RescaleValues` module isn't ready yet :(")


class Resize(nn.Module):
    """
    A PyTorch module that resizes the input tensor.
    """
    def __init__(self):
        """
        Initialize the `Resize` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resize` module.
        """
        raise NotImplementedError("The `Resize` module isn't ready yet :(")


class SoftQuantize(nn.Module):
    """
    A PyTorch module that maps continuious values to discrete bins.

    This module maps continuious values to discrete bins while retaining some smoothness/continuity
    which is parametrized by a softening parameter.
    """
    def __init__(self):
        """
        Initialize the `SoftQuantize` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `SoftQuantize` module.
        """
        raise NotImplementedError("The `SoftQuantize` module isn't ready yet :(")


# TODO: I disagree with putting this in neurite/torch/layers.py.
# I would prefer making a loss module.
class MSE(nn.Module):
    """
    A PyTorch module that calculates the mean squared error.
    """
    def __init__(self):
        """
        Initialize the `MSE` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `MSE` module.
        """
        raise NotImplementedError("The `MSE` module isn't ready yet :(")
