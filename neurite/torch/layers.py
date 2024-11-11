"""
Layers for the neurite project, written in PyTorch.

Citation
--------
If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018. https://arxiv.org/abs/1903.03148

License
-------
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
    "DrawImage",
    "SpatiallySparse_Dense",
    "LocalBias",
    "LocalLinear",
    "LocallyConnected3D",
    "LocalCrossLinear",
    "LocalCrossLinearTrf",
    "LocalParamLayer",
    "LocalParamWithInput",
    "MeanStream",
    "CovStream",
    "FFT"
    ]

from typing import Optional, Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from . import utils


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

        Returns
        -------
        torch.Tensor
            Negated tensor.
        """
        # Negate the tensor and return it.
        return -input_tensor


class RescaleValues(nn.Module):
    """
    A PyTorch module that rescales the values of the input tensor.
    """
    def __init__(self, scale_factor: float):
        """
        Initialize the `RescaleValues` module.

        Parameters
        ----------
        scale_factor : float
            Factor by which to rescale the values of the input tensor.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RescaleValues` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor to be resacled.

        Returns
        -------
        torch.Tensor
            Rescaled tensor.
        """
        raise input_tensor * self.scale_factor


class Resize(nn.Module):
    """
    A PyTorch module that resizes the input tensor.
    """
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
    ):
        """
        Initialize the `Resize` module.

        Parameters
        ----------
        size : int or Tuple[int, int], optional
            The desired output size. If None, uses `scale_factor`.
        scale_factor : float or Tuple[float, float], optional
            Scaling factor for resizing. If None, uses `size`.
        mode : str, default="nearest"
            Interpolation mode (e.g., "nearest", "bilinear").
        align_corners : bool, optional
            Alignment for "linear", "bilinear", or "trilinear" modes.
        recompute_scale_factor : bool, optional
            If True, recomputes the scale factor for interpolation.
        antialias : bool, default=False
            Applies anti-aliasing if `scale_factor` < 1.0.
        """
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resize` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be resized.

        Returns
        -------
        torch.Tensor
            The resized tensor.
        """
        resized_tensor = F.interpolate(
            input=input_tensor,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )
        raise resized_tensor


class SoftQuantize(nn.Module):
    """
    A PyTorch module that maps continuous values to discrete bins.

    This module maps continuous values to discrete bins while retaining some smoothness/continuous
    which is parametrized by a softening parameter. It is especially useful in the context of
    machine learning, where it is desirable to have a differentiable version of a quantized
    quantity, allowing for backprop. Hard quantization is non-differentiable and creates gradients
    of zero, making gradient-based optimization impossible.

    Parameters
    ----------
    nb_bins : int, optional
        The number of discrete bins to softly quantize the input values into. By default 16
    softness : float, optional
        The softness factor for quantization. A higher value gives smoother quantization.
        By default 1.0
    min_clip : float, optional
        Clip data lower than this value before calculating bin centers. By default -float('inf')
    max_clip : float, optional
        Clip data higher than this value before calculating bin centers. By default float('inf')
    return_log : bool, optional
        Optionally return the log of the softly quantized tensor. By default False

    Examples
    --------
    >>> import torch
    >>> import matplotlib.pyplot as plt
    # Make a random 3D tensor with zero mean and unit variance.
    >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
    # Initialize the SoftQuantize instance.
    >>> soft_quantizer = SoftQuantize(nb_bins=4, softness=0.5)
    # Apply the SoftQuantize instance to the input tensor
    >>> softly_quantized_tensor = soft_quantizer(input_tensor)
    # Visualize the softly quantized tensor.
    >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
    """
    def __init__(
        self,
        nb_bins: int = 16,
        softness: float = 1.0,
        min_clip: float = -float('inf'),
        max_clip: float = float('inf'),
        return_log: bool = False
    ):
        """
        Initialize the `SoftQuantize` module.

        Parameters
        ----------
        nb_bins : int, optional
            The number of discrete bins to softly quantize the input values into. By default 16
        softness : float, optional
            The softness factor for quantization. A higher value gives smoother quantization.
            By default 1.0
        min_clip : float, optional
            Clip data lower than this value before calculating bin centers. By default -float('inf')
        max_clip : float, optional
            Clip data higher than this value before calculating bin centers. By default float('inf')
        return_log : bool, optional
            Optionally return the log of the softly quantized tensor. By default False
        """
        super().__init__()
        self.nb_bins = nb_bins
        self.softness = softness
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.return_log = return_log

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `SoftQuantize` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to softly quantize.

        Returns
        -------
        torch.Tensor
            Softly quantized tensor with the same dimensions as `input_tensor`.
        """
        return utils.soft_quantize(
            input_tensor=input_tensor,
            nb_bins=self.nb_bins,
            softness=self.softness,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            return_log=self.return_log
        )


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

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `MSE` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor representing the model's prediction(s).
        target_tensor : torch.Tensor
            The target or ground truth values.

        Returns
        -------
        torch.Tensor
            The mean squared error between `input_tensor` and `target_tensor`.
        """
        return utils.mse_loss(input_tensor=input_tensor, target_tensor=target_tensor)


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
        raise NotImplementedError("The `DrawImage` module isn't ready yet :(")


# TODO: This is not how code should be modularized/encapsulated. This should go in another package.
#########################################################
# Sparse layers
#########################################################


class SpatiallySparse_Dense(nn.Module):
    """
    A Densely connected layer with sparse observations.
    """
    def __init__(self):
        """
        Initialize the `SpatiallySparse_Dense` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `SpatiallySparse_Dense` module.
        """
        raise NotImplementedError("The `SpatiallySparse_Dense` module isn't ready yet :(")


# TODO: Move to another package. Encapsulate code better.
#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################


class LocalBias(nn.Module):
    """
    A PyTorch module that applies biases to tensor elements independently.
    """
    def __init__(self):
        """
        Initialize the `LocalBias` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalBias` module.
        """
        raise NotImplementedError("The `LocalBias` module isn't ready yet :(")


# TODO: Make into AffineLayer and generalize to nD.
class LocalLinear(nn.Module):
    """
    A PyTorch module that applies linear transformations to tensor elements independently.
    """
    def __init__(self):
        """
        Initialize the `LocalLinear` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalLinear` module.
        """
        raise NotImplementedError("The `LocalLinear` module isn't ready yet :(")


class LocallyConnected3D(nn.Module):
    """
    A PyTorch module for unshared convolutions.

    Notes
    -----
    - The `LocallyConnected3D` layer works similarly to the `Conv3D` layer, except that weights are
    unshared, that is, a different set of filters is applied at each different patch of the input.
    """
    def __init__(self):
        """
        Initialize the `LocallyConnected3D` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocallyConnected3D` module.
        """
        raise NotImplementedError("The `LocallyConnected3D` module isn't ready yet :(")


class LocalCrossLinear(nn.Module):
    """
    A PyTorch module that ...
    """
    def __init__(self):
        """
        Initialize the `LocalCrossLinear` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalCrossLinear` module.
        """
        raise NotImplementedError("The `LocalCrossLinear` module isn't ready yet :(")


class LocalCrossLinearTrf(nn.Module):
    """
    A PyTorch module that ...
    """
    def __init__(self):
        """
        Initialize the `LocalCrossLinearTrf` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalCrossLinearTrf` module.
        """
        raise NotImplementedError("The `LocalCrossLinearTrf` module isn't ready yet :(")


class LocalParamLayer(nn.Module):
    """
    A PyTorch layer composed of a single unque parameter for each element in pixel/voxel space.

    References
    ----------
    - https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/SharedWeight.py
    - https://github.com/keras-team/keras/blob/ee02d256611b17d11e37b86bd4f618d7f2a37d84/keras/
    engine/input_layer.py
    """
    def __init__(self):
        """
        Initialize the `LocalParamLayer` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalParamLayer` module.
        """
        raise NotImplementedError("The `LocalParamLayer` module isn't ready yet :(")


class LocalParamWithInput(nn.Module):
    """
    A PyTorch module that ...
    """
    def __init__(self):
        """
        Initialize the `LocalParamWithInput` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `LocalParamWithInput` module.
        """
        raise NotImplementedError("The `LocalParamWithInput` module isn't ready yet :(")


class MeanStream(nn.Module):
    """
    A PyTorch module that ...
    """
    def __init__(self):
        """
        Initialize the `MeanStream` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `MeanStream` module.
        """
        raise NotImplementedError("The `MeanStream` module isn't ready yet :(")


class CovStream(nn.Module):
    """
    A PyTorch module that ...
    """
    def __init__(self):
        """
        Initialize the `CovStream` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `CovStream` module.
        """
        raise NotImplementedError("The `CovStream` module isn't ready yet :(")


# TODO: Encapsulate better. Move to a different package.
##########################################
# FFT Layers
##########################################


class FFT(nn.Module):
    """
    A PyTorch module that applies the fast forier transform to a tensor.
    """
    def __init__(self):
        """
        Initialize the `FFT` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `FFT` module.
        """
        raise NotImplementedError("The `FFT` module isn't ready yet :(")
