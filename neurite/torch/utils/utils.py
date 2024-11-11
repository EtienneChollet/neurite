"""
Utilities for the neurite project.

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
    'soft_quantize'
]

import torch
import torch.nn.functional as F


def soft_quantize(
    input_tensor: torch.Tensor,
    nb_bins: int = 16,
    softness: float = 1.0,
    min_clip: float = -float('inf'),
    max_clip: float = float('inf'),
    return_log: bool = False
):
    """
    This function softly maps continuous values to discrete bins while retaining smoothness,
    controlled by the `softness` parameter.

    This function is used to discretize continuous values into bins while preserving some continuity
    or smoothness in the data. It is especially useful in the context of machine learning, where it
    is desirable to have a differentiable version of a quantized quantity, allowing for backprop.
    Hard quantization is non-differentiable and creates gradients of zero, making gradient-based
    optimization impossible.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to softly quantize.
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

    Returns
    -------
    torch.Tensor
        Softly quantized tensor with the same dimensions as `input_tensor`.

    Examples
    --------
    >>> # Make a random 3D tensor with zero mean and unit variance.
    >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
    # Compute the softly quantized tensor with a low softness to approximate (and visualize) a
    # pseudo-hard quantization.
    >>> softly_quantized_tensor = soft_quantize(input_tensor, nb_bins=4, softness=0.0001)
    """
    # Invert softness
    softness = 1 / softness

    # Optionally clip `input_tensor`
    input_tensor.clip_(min_clip, max_clip)

    # Get the bin centers
    bin_centers = torch.linspace(
        start=input_tensor.min(),
        end=input_tensor.max(),
        steps=nb_bins,
        device=input_tensor.device
    )

    # Compute the distance between each element in `input_tensor` and the centers of the bins.
    # The resultant has `nb_bins` channels in the last dimension, each corresponding to the distance
    # between that element's intensity (in pixel/voxel space) to the center of each of the bins.
    distances_to_bin_centers = torch.abs(input_tensor.unsqueeze(-1) - bin_centers)

    # Apply softmax along last dimension
    softly_quantized = F.softmax(-softness * distances_to_bin_centers, dim=-1)

    # Compute the softly quantized value by averaging bin centers weighted by softmax values
    softly_quantized = (softly_quantized * bin_centers).sum(dim=-1)

    # Optionally convert to log domain
    if return_log:
        softly_quantized.log_()
    return softly_quantized
