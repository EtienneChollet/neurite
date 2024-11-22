"""
Augmentation Tools

This module provides a set of augmentation tools designed to enhance, randomize, and distort tensor
data.

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
    'random_crop',
]

from typing import Union, Tuple, List
import torch
from neurite.torch.random import RandInt, Sampler, Bernoulli, Uniform
from neurite.torch.utils import utils


def random_crop(
    input_tensor: torch.Tensor,
    crop_proportion: Union[Sampler, float] = 0.5,
    prob: Union[Sampler, float] = 1,
    forbidden_dims: Union[Tuple, List] = (0, 1),
    seed: Union[Sampler, int] = None,
):
    """
    Randomly crops the input tensor along specified dimensions based on a given proportion and
    probability.

    This function randomly selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    and crops each independently by a proportion that is randomly drawn from a distribution. The
    proportion to crop can either be fixed or sampled from a specified distribution. Each allowed
    dimension has a probability `prob` of being cropped based on the results of independent
    Bernoulli trials.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to be randomly cropped. It is assumed to have batch and channel dimensions.
    crop_proportion : Union[Sampler, float], optional
        The proportion that is randomly cropped from any allowed dimension. By default 0.5
        - If a `float` is provided, it represents the maximum proportion (0 to 1) to crop,
        sampled from independent uniform distributinos for each allowed dimension. A value of `0.5`
        means up to 50% of each dimension can be cropped.
        - If a `Sampler` is provided, cropped proportions are dynamically sampled based on the
        specified distribution
    prob : Union[Sampler, float], optional
        The probability of cropping each allowed dimension. By default 1.0
        - If a `float` is provided, it's used as a fixed probability for all eligible dimensions.
        - If a `Sampler` is provided, probabilities are dynamically generated for each dimension.
    forbidden_dims : Union[Tuple[int, ...], List[int]], optional
        Dimensions that should never be cropped. By defult `(0, 1)` (batch and channel dimensions)
    seed : Union[Sampler, int], optional
        A random seed or sampler to control the randomness of cropping operations. If provided, it
        ensures reproducibility of the cropping. Defaults to `None`.

    Returns
    -------
    torch.Tensor
        The randomly cropped tensor with the same number of dimensions as `input_tensor`.
        The size of each cropped dimension is reduced based on the sampled `crop_proportion`.

    Examples
    --------
    >>> tensor = torch.randn(2, 3, 64, 64)
    >>> # Cropping up to 80% of each dimension, with each having a 50% chance of being cropped
    >>> cropped_tensor = random_crop(tensor, crop_proportion=0.8, prob=0.5)
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 51, 51])

    >>> # Cropping between 25% to 75% of each dimension
    >>> cropped_tensor = random_crop(tensor, crop_proportion=Uniform(0.25, 0.75))
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 45, 45])

    >>> # Specifying forbidden dimensions (e.g., not cropping the last dimension)
    >>> cropped_tensor = random_crop(tensor, forbidden_dims=[0, 1, 3])
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 64, 32])
    """
    # Initialize random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    # Calculate the list of allowable dimensions
    allowed_dims = [x for x in range(input_tensor.dim()) if x not in forbidden_dims]
    # If `crop_proportion` is float, interpret it as upper bound of uniform distribution.
    crop_sampler = Uniform.make(utils.make_range(0, crop_proportion))
    # If prob is a sampler, sample from it
    if isinstance(prob, Sampler):
        prob = prob()
    # Make prob into a Bernoulli distribution
    prob = Bernoulli.make(prob)
    # Make empty list of slices which we will modufy
    slices = [slice(None)] * input_tensor.dim()
    # I think `translation_min` will always be zero. Keep it as such.
    translation_min = 0
    # Iterate through each dimension and make croppings for them independently.
    for dim in allowed_dims:
        # Decide if we are to crop the current dimension
        if bool(prob):
            # Determine the size of this dimension
            dim_size = input_tensor.shape[dim]
            # Sample a random proportion of the dimension to crop and convert it to a point along
            # the axis.
            crop_size = round((1 - crop_sampler()) * dim_size)
            # Calculate the maximum translation to avoid going out of bounds.
            translation_max = dim_size - crop_size
            # Prevent errors in the case of no translation
            if translation_min != translation_max:
                # Sample a valid translation (can't be out of bounds!)
                translation = RandInt(translation_min, translation_max)()
            else:
                translation = 0
            # Make the slice and override the current (presumably null) slice.
            slices[dim] = slice(translation, crop_size + translation)
    return input_tensor[slices]
