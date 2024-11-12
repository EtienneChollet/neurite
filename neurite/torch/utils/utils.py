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
    'soft_quantize',
    'mse_loss',
    'create_gaussian_kernel',
    'gaussian_smoothing',
    'sample_bernoulli_distribution',
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
) -> torch.Tensor:
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
    >>> import torch
    >>> import matplotlib.pyplot as plt
    # Make a random 3D tensor with zero mean and unit variance.
    >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
    # Compute the softly quantized tensor with a low softness to approximate (and visualize) a
    # pseudo-hard quantization.
    >>> softly_quantized_tensor = soft_quantize(input_tensor, nb_bins=4, softness=0.5)
    # Visualize the softly quantized tensor.
    >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
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


def mse_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) between the predicted and target values.

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

    Examples
    --------
    >>> import torch
    # Input tensor with zero mean, unit variance
    >>> input_tensor = torch.randn((1, 16, 16, 16))
    # Target tensor with zero mean, unit variance
    >>> target_tensor = torch.randn((1, 16, 16, 16))
    # Calculate loss
    >>> loss = mse_loss(input_tensor, target_tensor)
    # Print loss (should be approximately 2.0)
    >>> print(loss)
    """
    return torch.mean((input_tensor - target_tensor) ** 2)


def create_gaussian_kernel(kernel_size: int = 3, sigma: float = 1, ndim: int = 3):
    """
    Create a {1D, 2D, 3D} Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of Gaussian kernel. Default is 3.
    sigma : float
        Standard deviation of the Gaussian kernel. Default is 1.
    ndim : int
        Dimensionality of the gaussian kernel. Default is 3.

    Returns
    -------
    torch.Tensor
        Tensor representing the {1D, 2D, 3D} Gaussian kernel with batch and channel dimensions.

    Examples
    --------
    >>> import torch
    # Make the kernel!
    >>> gaussian_kernel = create_gaussian_kernel(3, 1, 3)
    # Print shape (should have batch and channel dimensions)
    >>> gaussian_kernel.shape()
    torch.Size([1, 1, 3, 3, 3])
    """
    # Create a coordinate grid centered at zero
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    grid = torch.stack(torch.meshgrid([coords] * ndim), -1)

    # Calculate the Gaussian function
    kernel = torch.exp(-((grid ** 2).sum(-1) / (2 * sigma ** 2)))

    # Normalize the kernel so that the sum of all elements is 1
    kernel = kernel / kernel.sum()

    # Reshape to 5D tensor for conv3d
    kernel = kernel.view(1, 1, *([kernel_size] * ndim))
    return kernel


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1,
) -> torch.Tensor:
    """
    Applies Gaussian smoothing to the {1D, 2D, 3D} input tensor based on the given kernel size and
    sigma.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor, assumed to be 1D, 2D, or 3D.
    kernel_size : int, optional
        Size of the Gaussian kernel, default is 3.
    sigma : float, optional
        Standard deviation of the Gaussian kernel, default is 1.

    Returns
    -------
    smoothed_tensor : torch.Tensor
        The smoothed tensor.

    Examples
    --------
    >>> import torch
    # Make a random input tensor
    >>> input_tensor = torch.rand(1, 1, 16, 16, 16)
    # Smooth it
    >>> smoothed_tensor = gaussian_smoothing(input_tensor)
    """
    # Infer dimensionality in voxel/pixel space. Squeeze to remove batch and/or channel dims.
    ndim = input_tensor.squeeze().dim()

    # Initialize the gaussian kernel
    gaussian_kernel = create_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        ndim=ndim
    )

    # Calculate padding size
    padding = kernel_size // 2
    # Make the padding symmetric and 
    padding = torch.tensor(padding).repeat(ndim * 2)
    # Convert to tuple (F.pad takes a tuple of ints, not tensors)
    padding = tuple(padding.tolist())
    # Pad `input_tensor`
    padded_input_tensor = F.pad(input_tensor, padding, mode='reflect')

    # Make dictionary for the different convolution dimensionalities
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]
    # Apply the smoothig operation
    smoothed_tensor = conv_fn(
        input=padded_input_tensor,
        weight=gaussian_kernel,
        padding=0,
    )
    return smoothed_tensor


def sample_bernoulli_distribution(p: float = 0.5, shape: tuple = (1,)):
    """
    Sample from a Bernoulli distribution with a specified probability and shape.

    Parameters
    ----------
    p : float, optional
        Probability of realizing a success (i.e., the probability of a 1) from the Bernoulli
        distribution. By default, 0.5. Must be in the range [0, 1].
    shape : tuple, optional
        Shape of the output tensor, specifying the number of independent Bernoulli trials. Each
        entry represents the dimensions of the resulting tensor. By default, (1).

    Returns
    -------
    bernoulli_result: torch.Tensor
        A tensor of Bernoulli-distributed random samples with values of 0 or 1, representing results
        of independent Bernoulli trials.

    Examples
    --------
    >>> # Generate samples from the Bernoulli distribution
    >>> samples = sample_bernoulli(p=0.25, shape=(1, 32, 32, 32))
    >>> # Mean (expectation) should be ~=`p`
    >>> print(samples.mean())
    tensor(0.2471)
    """
    # Make sampling domain.
    # Each element in this tensor represents the probability of realizing a 1.
    sampling_domain = torch.tensor(p).repeat(shape)
    # Sample from the bernoulli distribution
    bernoulli_result = torch.bernoulli(sampling_domain)
    return bernoulli_result
