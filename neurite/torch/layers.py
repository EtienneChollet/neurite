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
    "MSE",
    "GaussianBlur",
    "Subsample",
    "RandomCrop",
    "RandomClip",
    "RandomGamma",
    "RandomIntensityLookup",
    "RandomClearLabel",
    "DrawImage"
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


# TODO: I disagree with putting this in neurite/torch/layers.py. Can we move to a loss module?
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


# TODO: Move to an augmentation package/repo?
class GaussianBlur(nn.Module):
    """
    A PyTorch module to blur a tensor by convolving it with a Gaussian kernel.
    """
    def __init__(self):
        """
        Initialize the `GaussianBlur` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `GaussianBlur` module.
        """
        raise NotImplementedError("The `GaussianBlur` module isn't ready yet :(")


# TODO: Move to an augmentation package/repo?
class Subsample(nn.Module):
    """
    A PyTorch module to subsample a tensor.

    This module subsamples the input tensor by a factor `f` (stride) along one dimension using the
    nearest-neighbor algorithm. Options exist to squsequently downsample the tensor to restore it's
    original shape.
    """
    def __init__(self):
        """
        Initialize the `Subsample` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Subsample` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `Subsample` module isn't ready yet :(")


# TODO: Move to an augmentation package/repo?
class RandomCrop(nn.Module):
    """
    A PyTorch module that randomly crops the input tensor to a particular field of view.

    Randomly crop a tensor by multiplying with a spatially continuious binary mask (as opposed to
    brenouli sampling).
    """
    def __init__(self):
        """
        Initialize the `RandomCrop` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomCrop` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `RandomCrop` module isn't ready yet :(")


# TODO: Move to an augmentation package/repo?
class RandomClip(nn.Module):
    """
    A PyTorch module that randomly clips elements outside the bounds.

    Randomly select elements within a tensor using Brenouli sampling, and clip values falling
    outside the (exclusive) range [`clip_min`, `clip_max`].
    """
    def __init__(self):
        """
        Initialize the `RandomClip` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClip` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `RandomClip` module isn't ready yet :(")


# TODO: Move to an augmentation package/repo?
class RandomGamma(nn.Module):
    """
    A PyTorch module that applies a random gamma transformation to a tensor.

    Applies a gamma (exponential) transformation to the elements of the input tensor by drawing the
    exponentiation factor from a uniform distribution.
    """
    def __init__(self):
        """
        Initialize the `RandomGamma` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomGamma` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `RandomGamma` module isn't ready yet :(")


# TODO: Move to an augmentation package/repo?
class RandomIntensityLookup(nn.Module):
    """
    A PyTorch module to augment the contrast of a single-channel tensor.

    Compute a smoothly varying lookup table to map the original single-channel tensor (usually a
    greyscale image) to a tensor with a new contrast.
    """
    def __init__(self):
        """
        Initialize the `RandomIntensityLookup` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomIntensityLookup` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `RandomIntensityLookup` module isn't ready yet :(")


# TODO: I suggest renaming this to BrenouliMasker
# TODO: Move to an augmentation package/repo?
class RandomClearLabel(nn.Module):
    """
    A PyTorch module that applies a Brenouli mask to the input tensor.

    Compute a Brenouli mask from the shape of the input tensor and apply it to return a tensor whose
    elements have been dropped out randomly.
    """
    def __init__(self):
        """
        Initialize the `RandomClearLabel` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClearLabel` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `RandomClearLabel` module isn't ready yet :(")


# TODO: Rename something more intuitive like `LabelsToIntensity`?
# TODO: Move to an augmentation package/repo?
class DrawImage(nn.Module):
    """
    A PyTorch module that generates an image from a label map by uniformly sampling a random
    intensity for each label.
    """
    def __init__(self):
        """
        Initialize the `DrawImage` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `DrawImage` module.
        """
        # Negate the tensor and return it.
        raise NotImplementedError("The `DrawImage` module isn't ready yet :(")
