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
    "register_init_arguments",
    "BaseTransform",
    "TransformList",
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
    "SampleImageFromLabels",
    "DrawImage",
    "Norm",
    "Activation",
    "LocalParamLayer",
    "LocalParamWithInput",
    "MeanStream",
    "CovStream",
    "FFT",
]

import inspect
import warnings
from typing import Optional, Union, Tuple, Callable, List, Type
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from . import utils
from ..torch.random import Sampler, Bernoulli, Fixed, Uniform, Normal


def register_init_arguments(func: Callable) -> Callable:
    """
    Decorator to register a function's (typically __init__) arguments into the instance's
    `arguments` dictionary and set them as individual attributes.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original __init__ method
        result = func(self, *args, **kwargs)

        # Initialize the arguments dictionary if it doesn't exist
        if not hasattr(self, 'arguments'):
            self.arguments = {}

        # Get the function's signature
        sig = inspect.signature(func)

        # Bind the passed arguments to the signature
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Extract parameters excluding 'self'
        params = {k: v for k, v in bound.arguments.items() if k != 'self'}

        # Unpack 'theta' if present
        if 'theta' in params:
            theta = params.pop('theta')
            for key, value in theta.items():
                # If the value is a Sampler instance, store its arguments recursively
                if isinstance(value, Sampler):
                    self.arguments[key] = value.serialize()  # Or value.arguments for direct args
                else:
                    self.arguments[key] = value
                setattr(self, key, value)  # Set as attribute

        # Register the individual arguments and set them as attributes
        for key, value in params.items():
            self.arguments[key] = value
            setattr(self, key, value)  # Set as attribute

        return result
    return wrapper


class BaseTransform(nn.Module):
    """
    Base class for tensor transformations, containing arbitrary parameters `theta` and
    serialization logic.

    This class provides a foundation for implementing customizable and easily communicable
    transformations by allowing arbitrary parameters (`theta`) to be passed during
    initialization. It also supports serialization of its state, enabling metadata extraction
    and facilitating reconstruction of instances.

    Attributes
    ----------
    share : Union[bool, str]
        Strategy for sharing transformations.
        - `True`: Apply the same transformation to the entire tensor.
        - `'channel'`: Apply the same transformation across channels for each batch element.
        - `'batch'`: Apply the same transformation across the batch for each channel.
        - `False`: Apply distinct transformations for each channel of each batch element.
    theta : Dict[str, Any]
        Arbitrary parameters for the transformation.

    Notes
    -----
    - Subclasses of `BaseTransform` must define a `transform()` method, as opposed to the nn.Module
    convention of `forward()` method.
    - Currently, `BaseTransform` is only able to handle transformations that involve a single
    `input_tensor`, so paired x-y transformations are not possible (yet).
    - Subclasses of `BaseTransform` must pass all initialization arguments into the
    `super().__init__()` constructor. This allows the arguments to be registered to theta, and thus
    saved upon serialization.
    """

    @register_init_arguments
    def __init__(self, share: Union[bool, str] = True, **theta):
        """
        Initialize the transformation with arbitrary parameters.

        Parameters
        ----------
        share : Union[bool, str], optional
            Sharing strategy, by default `True`.
            - `True`: Share across all dimensions.
            - `'channel'`: Share across the channel dimension.
            - `'batch'`: Share across the batch dimension.
            - `False`: No sharing (individual transformations).
        **theta : Any
            Arbitrary keyword arguments representing transformation parameters.
        """
        super().__init__()
        self.theta = theta
        self.share = share

    def serialize(self) -> dict:
        """
        Serializes the object's state into a dictionary.

        This method captures key attributes of the object and metadata about its class and module
        for purposes such as taxonomy, reconstruction, or debugging.

        Returns
        -------
        dict
            A dictionary containing the following fields:
            - `qualname` (str): Fully qualified name of the class, useful for reconstructing the
               object.
            - `parent` (str): Name of the immediate parent class, useful for hierarchical taxonomy
               or debugging.
            - `module` (str): Module name where the class is defined, for locating and
               reconstructing the object.
            - `arguments` (dict): The arbitrary parameters (`theta`) passed during initialization.
        """
        state_dict = {
            # The qualified name of the class (for reconstruction purposes)
            'qualname': self.__class__.__name__,
            # Parent class, for more broad taxonomy/snapshot view
            'parent': type(self).__bases__[0].__name__,
            # The module that the sample may be found in (and reconstructed from)
            'module': self.__module__,
            # The sampler's parameters
            'arguments': self.arguments,
        }
        return state_dict

    def transform(self, input_tensor: torch.Tensor):
        """
        Define and apply the transformation to the input tensor.

        This method should be overridden by subclasses to implement specific transformations.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to transform.

        Returns
        -------
        torch.Tensor
            The transformed tensor.
        """
        raise NotImplementedError("`BaseTransform` is not a valid transform :(")

    def forward(self, input_tensor: torch.Tensor):
        """
        Forward pass of `BaseTransform`

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to transform.

        Returns
        -------
        torch.Tensor
            The transformed tensor.
        """
        # Case 1: Apply the same transformation to the entire tensor
        if self.share is True:
            return self.transform(input_tensor)

        # Case 2: Apply different transformations to each batch element, same across channels
        elif self.share == 'channel':
            # Initialize a list to hold transformed batch elements
            transformed = []
            for i in range(input_tensor.shape[1]):  # Loop over batch elements
                # Select the i-th batch element
                batch_element = input_tensor[:, i].unsqueeze(0).unsqueeze(0)
                # Apply the transformation
                transformed_element = self.transform(batch_element)[0, 0]
                transformed.append(transformed_element)
            # Stack the transformed elements back into a tensor
            return torch.stack(transformed, dim=1)

        # Case 3: Apply different transformations to each channel, same across batch
        elif self.share == 'batch':
            transformed_channels = []
            for minibatch in range(input_tensor.shape[0]):
                channel = input_tensor[minibatch].unsqueeze(0)
                transformed_channel = self.transform(channel)[0]
                transformed_channels.append(transformed_channel)
            # Stack the transformed channels back into a tensor along channel dimension
            return torch.stack(transformed_channels, dim=0)

        # Case 4: Apply different transformations to each channel of each batch element
        elif self.share is False:
            # Initialize a list to hold transformed batch elements
            transformed = []
            for i in range(input_tensor.shape[0]):
                # Initialize a list to hold transformed channels for the i-th batch element
                transformed_channels = []
                for c in range(input_tensor.shape[1]):
                    # Select the (i, c)-th channel
                    channel = input_tensor[i, c].unsqueeze(0).unsqueeze(0)
                    # Apply the transformation
                    transformed_channel = self.transform(channel)[0][0]
                    transformed_channels.append(transformed_channel)
                # Stack the transformed channels back into a tensor for the i-th batch element
                transformed_element = torch.stack(transformed_channels, dim=0)
                transformed.append(transformed_element)
            # Stack the transformed batch elements back into a tensor
            return torch.stack(transformed, dim=0)
        else:
            raise AttributeError(f"Could not interpret share type: {self.share}")


class TransformList(nn.Module):
    """
    A container for managing, serializing, and applying a sequence of transformations
    that inherit from `BaseTransform`. Each transformation is applied with a global or
    independent probability drawn from iid Bernoulli trials.
    """

    def __init__(
        self,
        transforms: nn.ModuleList,
        probabilities: Union[list, float, int] = 1
    ):
        """
        Initialize the `TransformList` with a list of transformations and their corresponding
        probabilities.

        Parameters
        ----------
        transforms : nn.ModuleList
            A list of transformations inheriting from `BaseTransform`.
        probabilities : list
            A single probability or list of probabilities for applying each transform. If a single
            probability is provided, it is applied uniformly to all transforms. By default 1,
            meaning all transforms are always applied.

        Examples
        --------
        ### Initializing with transforms
        >>> transforms = TransformList([GaussianBlur(), Resample()])
        >>> # Get serialized state of list and print qualnames
        >>> serialized_state = transforms.serialize()
        >>> [i['qualname'] for i in serialized_state]
        ['GaussianBlur', 'Resample']

        ### Appending modules to the list
        >>> # Now let's append a module to the list
        >>> transforms.append(GaussianBlur())
        >>> # Print each qualname now
        >>> serialized_state = transforms.serialize()
        >>> [i['qualname'] for i in serialized_state]
        ['GaussianBlur', 'Resample', 'GaussianBlur']

        ### Modify a tensor with the transforms
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> transformed_tensor = transforms(input_tensor)

        ### Apply transforms with given probabilities
        >>> # e.g. we want to apply gaussian blurring a lot, but resample much less.
        >>> transforms = [GaussianBlur(), Resample()]
        >>> probabilities = [0.95, 0.1]
        >>> transforms = TransformList(transforms, probs)
        """
        super().__init__()
        # Assign instance attribute
        self.transforms = transforms
        # If global probability is defined, make it into list equal to the length of the transforms
        if not isinstance(probabilities, (list | tuple)):
            # Will resilt in a list of identically distributed Bernoulli trials
            probabilities = [probabilities] * len(transforms)
        # If probabilities is a list but not length is not equal to transforms, we have a problem!
        elif isinstance(probabilities, (list | tuple)):
            assert len(transforms) == len(probabilities), (
                "Transforms and probabilities must have the same length."
            )
        # Make a list of independent Bernoulli trials associated with the respective transform
        self.apply_transform_probs = [Bernoulli(p=p) for p in probabilities]

    def serialize(self) -> list:
        """
        Serializes the list of transformations into a list of dictionaries, including their applying
        probabilities.

        Returns
        -------
        list
            A list of dictionaries, each containing the serialized transform and its
            associated applying probability.
        """
        # Container that we will sequentially populate with serialized transfomations
        serialized_transforms = []
        # Iterate through the transformations and corresponding Bernoulli samplers
        for transform, apply_prob in zip(self.transforms, self.apply_transform_probs):
            if hasattr(transform, 'serialize') and callable(transform.serialize):
                serialized_transforms.append({
                    "transform": transform.serialize(),
                    "apply_probability": apply_prob.theta['p']
                })
            else:
                raise ValueError(
                    f"Transform {type(transform).__name__} does not support serialization."
                )
        return serialized_transforms

    def append(self, transform: nn.Module):
        """
        Add a transform to the list.

        Parameters
        ----------
        transform : nn.Module
            A transform inheriting from `BaseTransform`.
        """
        if isinstance(transform, BaseTransform):
            self.transforms.append(transform)
        else:
            raise TypeError(
                f"Transform must inherit from BaseTransform, got {type(transform)} instead."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Sequentially applies all transformations in the list with a given probability to the input
        tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be transformed.

        Returns
        -------
        torch.Tensor
            The transformed tensor.
        """
        for transform, apply_prob in zip(self.transforms, self.apply_transform_probs):
            # Sample a binary decision to apply a particular transform or not
            apply_transform = apply_prob().item()
            if apply_transform:
                input_tensor = transform(input_tensor)
        return input_tensor


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


class RescaleValues(BaseTransform):
    """
    A PyTorch module that rescales the values of the input tensor.
    """

    def __init__(self, scale_factor: Union[float, int, Sampler]):
        """
        Initialize the `RescaleValues` module.

        Parameters
        ----------
        scale_factor : float, int, or Sampler
            Factor (or sampler) by which to rescale the values of the input tensor.
        """
        super().__init__(
            scale_factor=Fixed.make(scale_factor),
        )

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
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
        return input_tensor * self.scale_factor()


class Resize(nn.Module):
    """
    A PyTorch module that resizes the input tensor.
    """
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
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

        Examples
        --------
        >>> # Define a tensor we'll use for resizing examples
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)

        ### Resizing with fixed `scale_factor`
        >>> transform = Resize(scale_factor=2)
        >>> resized_tensor = transform(input_tensor)
        >>> print(resized_tensor.shape)
        torch.Size([1, 1, 64, 64, 64])

        ### Resizing with a sampled `scale_factor`
        >>> transform = Resize(scale_factor=Uniform(0.5, 4))
        >>> resized_tensor = transform(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 74, 74, 74])

        ### Resizing to a specific size
        >>> transform = Resize(size=(96, 96, 96))
        >>> resized_tensor = transform(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 96, 96, 96])

        Notes
        -----
        - This class assumes the input tensor has batch and channel dimensions.
        - It is not possible to define `size` and `scale_factor` simultaneously. Only one can be
        defined for a given instatntiation of `Resize`.
        - When defining `size` do not include batch or channel dimensions, only spatial dims.
        """
        super().__init__()
        # Either scale factor or size must be defined. If neither is, make scale factor 1.
        if size is None and scale_factor is None:
            scale_factor = Fixed(1)
        elif scale_factor is not None:
            # Make a fixed if passed a single number. Maks sampler if passed sampler.
            scale_factor = Fixed.make(scale_factor)

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
            The input tensor to be resized. Must have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The resized tensor.
        """
        resized_tensor = F.interpolate(
            input=input_tensor,
            size=self.size,
            scale_factor=self.scale_factor(),
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )
        return resized_tensor


class SoftQuantize(BaseTransform):
    """
    A PyTorch module that maps continuous values to discrete bins.

    This module maps continuous values to discrete bins while retaining some smoothness/continuous
    which is parametrized by a softening parameter. It is especially useful in the context of
    machine learning, where it is desirable to have a differentiable version of a quantized
    quantity, allowing for backprop. Hard quantization is non-differentiable and creates gradients
    of zero, making gradient-based optimization impossible.
    """

    def __init__(
        self,
        nb_bins: Union[int, Sampler] = 16,
        softness: Union[float, int, Sampler] = 1.0,
        min_clip: Union[float, int, Sampler] = -float('inf'),
        max_clip: Union[float, int, Sampler] = float('inf'),
        return_log: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize the `SoftQuantize` module.

        Parameters
        ----------
        nb_bins : int or Sampler, optional
            The number of discrete bins to softly quantize the input values into. By default 16
        softness : float, int, or Sampler, optional
            The softness factor for quantization. A higher value gives smoother quantization.
            By default 1.0
        min_clip : float, int, or Sampler, optional
            Clip data lower than this value before calculating bin centers. By default -float('inf')
        max_clip : float, int, or Sampler, optional
            Clip data higher than this value before calculating bin centers. By default float('inf')
        return_log : bool, optional
            Optionally return the log of the softly quantized tensor. By default False

        Examples
        --------
        >>> # Make a random 3D tensor with zero mean and unit variance.
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Initialize the SoftQuantize instance.
        >>> soft_quantizer = SoftQuantize(nb_bins=4, softness=0.5)
        >>> # Apply the SoftQuantize instance to the input tensor
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> # Visualize the softly quantized tensor.
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])

        ### Softly quantizing with randomly sampled `nb_bins` and `softness` parameters
        >>> # Define `nb_bins` to sample a uniform int distribution, and `softness` a float dist
        >>> soft_quantizer = SoftQuantize(nb_bins=RandInt(3, 32), softness=Uniform(0.001, 10))
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
        """
        super().__init__(
            nb_bins=nb_bins,
            softness=softness,
            min_clip=min_clip,
            max_clip=max_clip,
            return_log=return_log,
            *args,
            **kwargs
        )

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
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


class GaussianBlur(BaseTransform):
    """
    A PyTorch module to blur a {1D, 2D, 3D} tensor by convolving it with a Gaussian kernel.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1,
        *args,
        **kwargs
    ):
        """
        Initialize the `GaussianBlur` module.

        Parameters
        ----------
        kernel_size : Sampler or int, optional
            Size of the Gaussian kernel, default is 3.
        sigma : float, int, or Sampler, optional
            Standard deviation of the Gaussian kernel, default is 1.
        """
        super().__init__(
            kernel_size=kernel_size,
            sigma=sigma,
            *args,
            **kwargs
        )

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `GaussianBlur` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor, assumed to be 1D, 2D, or 3D.

        Returns
        -------
        torch.Tensor
            The smoothed tensor.
        """
        return utils.gaussian_smoothing(
            input_tensor=input_tensor,
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )


class Resample(BaseTransform):
    """
    A PyTorch module to resample a tensor.

    This module resamples the input tensor by a factor of `stride` along the
    specified dimension by interleaving dropouts along it (keeping every `stride`th element).
    Optionally upsample the tensor after downsampling it to restore it to its original dimensions.
    """

    def __init__(
        self,
        upsample: bool = True,
        stride: Union[int, Tuple[int, int]] = 2,
        forbidden_dims: Tuple[int, int] = (1, 0),
        p: float = 0.5,
        max_concurrent_subsamplings: int = None,
        mode: str = 'nearest',
        *args,
        **kwargs
    ):
        """
        Initialize the `Resample` module.

        Parameters
        ----------
        upsample : bool, optional
            optionally upsample the subsampled tensor to the original dimensions of `input_tensor`.
            By default, True.
        stride : Sampler or int or tuple, optional
            The stride value to use when subsampling a given dimension. Can be an integer or
            a tuple corresponding to the range of strides to sample. By default, 2.
                - A stride of 1 does not result in any subsampling.
                - A stride of 2 will reduce the elements of the selected dimension by 1/2.
        forbidden_dims : list, optional
            A list of dimensions that should not be subsampled. If None, no dimensions
            are forbidden from subsampling. Default is (0, 1) to ignore batch and channel
            dimensions.
        p : float, optional
            The probability of selecting each dimension for subsampling. This probability
            is applied as an independent Bernoulli trial for each dimension. By default, 0.5.
        max_concurrent_subsamplings : int, optional
            The maximum number of dimensions that can be subsampled simultaneously. If
            None, the number of concurrent subsamplings is set to the number of dimensions
            in `input_tensor`. Default is None.
        mode : str, optional
            The interpolation mode to use for upsampling. By default None. Options (WRT spatial
            dimensions) include:
                - 'nearest' (default)
                - 'linear' (1D-only)
                - 'bilinear' (2D-only)
                - 'bicubic' (2D-only)
                - 'trilinear' (3D-only)
                - 'area'

        Examples
        --------
        ### Custom stride and only subsampling
        >>> # Initialize a random 3D tensor with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128, 128)
        >>> # Resample the tensor with random strides on the (inclusive) interval (2, 5)
        >>> resampled_tensor = Resample(upsample=False, stride=(2, 5))(input_tensor)
        >>> # Spatial dimensions be different
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 64, 128, 32])

        ### Custom stride with upsampling and trilinear interpolation
        >>> # Initialize a random 3D tensor with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128, 128)
        >>> # Resample the tensor with a stride upper bound of 6, trilinear interpolation, and with
        a final upsampling operation after downsampling.
        >>> resampled_tensor = Resample(stride=6, mode='trilinear')(input_tensor)
        >>> # Spatial dims should be the same (because the last operation is upsampling to original)
        >>> print(input_tensor.shape)
        torch.Size([1, 1, 128, 128, 128])
        """
        super().__init__(
            upsample=upsample,
            stride=stride,
            forbidden_dims=forbidden_dims,
            p=p,
            max_concurrent_subsamplings=max_concurrent_subsamplings,
            mode=mode,
            *args,
            **kwargs
        )

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resample` module.
        """
        # Store original spatial shape to restore dimensions in upsampling
        resampled_tensor = input_tensor
        original_spatial_shape = input_tensor.shape[2:]

        # Start by subsampling the input tensor
        resampled_tensor = utils.subsample_tensor_random_dims(
            input_tensor=resampled_tensor,
            stride=self.theta.get('stride'),
            forbidden_dims=self.theta.get('forbidden_dims'),
            p=self.theta.get('p'),
            max_concurrent_subsamplings=self.theta.get('max_concurrent_subsamplings')
        )

        # Optionally upsample the resuling subsampled tensor
        if self.upsample:
            # Apply upsampling
            resampled_tensor = utils.upsample_tensor(resampled_tensor, original_spatial_shape)

        return resampled_tensor


class Subsample(Resample):
    """
    @deprecated: Use `Resample` instead.
    A PyTorch module to subsample and upsample a tensor based on specified operations.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Subsample is deprecated and will be removed in future versions. "
            "Use ResampleModule instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class RandomCrop(BaseTransform):
    """
    A PyTorch module that randomly crops the input tensor to a particular field of view.

    This module randomly selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    and crops each independently by a proportion that is randomly drawn from a distribution. The
    proportion to crop can either be fixed or sampled from a specified distribution. Each allowed
    dimension has a probability `prob` of being cropped based on the results of independent
    Bernoulli trials.
    """

    def __init__(
        self,
        crop_proportion: Union[Sampler, float] = 0.5,
        prob: Union[Sampler, float] = 1,
        forbidden_dims: Union[Tuple, List] = (0, 1),
        seed: Union[int, Sampler] = None,
    ):
        """
        Initialize the `RandomCrop` module.

        Parameters
        ----------
        crop_proportion : Union[Sampler, float], optional
            The proportion that is randomly cropped from any allowed dimension. By default 0.5
            - If a `float` is provided, it represents the maximum proportion (0 to 1) to crop,
            sampled from independent uniform distributinos for each allowed dimension. A value of
            `0.5` means up to 50% of each dimension can be cropped.
            - If a `Sampler` is provided, cropped proportions are dynamically sampled based on the
            specified distribution
        prob : Union[Sampler, float], optional
            The probability of cropping each allowed dimension. By default 1.0
            - If a `float` is provided, it's used as a fixed probability for all eligible
            dimensions.
            - If a `Sampler` is provided, probabilities are dynamically generated for each
            dimension.
        forbidden_dims : Union[Tuple[int, ...], List[int]], optional
            Dimensions that should never be cropped. By defult `(0, 1)` (batch and channel
            dimensions)
        seed : Union[int, Sampler], optional
            A random seed or sampler to control the randomness of cropping operations. If provided,
            it ensures reproducibility of the cropping. Defaults to `None`.
        """
        super().__init__(
            crop_proportion=crop_proportion,
            prob=prob,
            forbidden_dims=forbidden_dims,
            seed=seed
        )

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomCrop` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to be randomly cropped. It is assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The tensor that has been randomly cropped.
        """
        return utils.random_crop(
            input_tensor=input_tensor,
            crop_proportion=self.crop_proportion,
            prob=self.prob,
            forbidden_dims=self.forbidden_dims,
            seed=self.seed
        )


class RandomClip(BaseTransform):
    """
    A PyTorch module that randomly clips tensor elements outside the bounds.
    """

    def __init__(
        self,
        clip_min: Union[float, int, Sampler] = 0,
        clip_max: Union[float, int, Sampler] = 1,
        clip_prob: Union[float, int, Sampler] = 0.5,
        seed: Union[int, Sampler] = None,
        *args,
        **kwargs
    ):
        """
        Initialize `RandomClip` with specified clipping bounds and sampling probability.

        Parameters
        ----------
        clip_min : Union[float, int, Sampler], optional
            The lower bound for clipping. Elements less than `clip_min` are set to `clip_min`.
            Defaults to 0.
        clip_max : Union[float, int, Sampler], optional
            The upper bound for clipping. Elements greater than `clip_max` are set to `clip_max`.
            Defaults to 1.
        clip_prob : Union[float, int, Sampler], optional
            Probability of applying the clipping transformation. Defaults to 0.5.
        seed : Union[int, Sampler], optional
            Seed for random number generation to ensure reproducibility. Defaults to None.

        Examples
        --------
        ### Initialize the `RandomClip` module and apply it to a tensor:
        >>> transform = RandomClip(clip_min=0.1, clip_max=0.9, clip_prob=0.5)
        >>> input_tensor = torch.randn(3, 3)
        >>> output_tensor = transform(input_tensor)
        >>> print(output_tensor)

        ### Use a sampler for dynamic clipping bounds:
        >>> from my_samplers import UniformSampler
        >>> transform = RandomClip(
                clip_min=UniformSampler(0, 0.5),
                clip_max=UniformSampler(0.5, 1.0)
            )
        >>> output_tensor = transform(input_tensor)
        >>> print(output_tensor)
        """
        super().__init__(
            clip_min=clip_min,
            clip_max=clip_max,
            clip_prob=clip_prob,
            seed=seed,
            *args,
            **kwargs
        ) 

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClip` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be clipped.

        Returns
        -------
        torch.Tensor
            The clipped tensor (if Bernoulli trial defined by parameter `clip_prob` is successful).
        """
        return utils.random_clip(
            input_tensor=input_tensor,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            clip_prob=self.clip_prob,
            seed=self.seed
        )


class RandomGamma(BaseTransform):
    """
    A PyTorch module that applies a random gamma transformation to a tensor.

    The gamma transformation adjusts the contrast of the input tensor by applying a non-linear
    operation. Specifically, each element in the tensor is raised to the power of `gamma`. This can
    enhance or diminish the contrast of the input data, making it a valuable augmentation tool for
    various deep learning tasks.
    """

    def __init__(
        self,
        gamma: Union[float, int, Sampler] = 1.0,
        prob: Union[float, int, Sampler] = 1.0,
        seed: Union[int, Sampler] = None,
    ):
        """
        Initialize the `RandomGamma` module.

        Parameters
        ----------
        gamma : Union[float, int, Sampler], optional
            The gamma value to apply for the transformation.
            - If a `float` is provided, it represents a fixed gamma value.
            - If a `Sampler` is provided, the gamma value is dynamically sampled from the specified
            distribution.
            By default `1.0`, which leaves the tensor unchanged.
        prob : Union[float, int, Sampler], optional
            The probability of applying the gamma transformation.
            - If a `float` is provided, it's used as a fixed probability for the transformation.
            - If a `Sampler` is provided, probabilities are dynamically generated for each
            invocation.
            Default is `1.0` (always apply).
        seed : Union[int, Sampler], optional
            A random seed or sampler to control the randomness of the gamma transformation. If
            provided, it ensures reproducibility of the transformation. Defaults to `None`.

        Examples
        --------
        ### Fixed gamma transformation
        >>> transform = RandomGamma(gamma=2.0, prob=1.0)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor = transform(tensor)
        >>> print(gamma_tensor)
        tensor([0.0625, 0.2500, 0.5625])

        ### Randomized gamma transformation with a range of gamma values
        >>> gamma_sampler = Uniform(0.5, 1.5)
        >>> transform = RandomGamma(gamma=gamma_sampler, prob=0.8)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor = transform(tensor)
        >>> print(gamma_tensor)
        tensor([0.1768, 0.5000, 0.8367])

        ### Applying gamma transformation with reproducibility
        >>> transform1 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> transform2 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor1 = transform1(tensor)
        >>> gamma_tensor2 = transform2(tensor)
        >>> print(torch.equal(gamma_tensor1, gamma_tensor2))
        True
        """
        super().__init__(gamma=gamma, prob=prob, seed=seed)

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomGamma` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to which the gamma transformation will be applied. It is assumed to
            have a range suitable for gamma correction (typically normalized between 0 and 1).

        Returns
        -------
        torch.Tensor
            The tensor after applying the gamma transformation. If the transformation is not applied
            (based on `prob`), the original `input_tensor` is returned unchanged.
        """
        return utils.random_gamma(
            input_tensor=input_tensor,
            gamma=self.gamma,
            prob=self.prob,
            seed=self.seed
        )


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


class RandomClearLabel(nn.Module):
    """
    A PyTorch module to randomly clear/erase regions from an image corresponding to randomly
    selected entities/continuious regions in a label map.

    This module identifies unique labels within the `label_tensor` and, based on a specified
    probability, designates regions of the `input_tensor` to be cleared (set to zero) corresponding
    to randomly selected labels. This can be used for tasks such as data augmentation, where certain
    labels are randomly omitted to simulate occlusions or missing annotations.

    Examples
    --------
    ### Clearing labels with a fixed probability
    >>> transform = RandomClearLabel(prob=0.75)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor = transform(input_tensor, label_tensor)
    >>> print(cleared_tensor)
    tensor([0.0000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000])

    ### Reproducibility with a seed
    >>> transform1 = RandomClearLabel(seed=32)
    >>> transform2 = RandomClearLabel(seed=32)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor1 = transform1(input_tensor, label_tensor)
    >>> cleared_tensor2 = transform2(input_tensor, label_tensor)
    >>> print(torch.equal(cleared_tensor1, cleared_tensor2))
    True
    """

    def __init__(
        self,
        prob: Union[float, int, Sampler] = 0.5,
        exclude_zero: bool = True,
        seed: int = None,
    ):
        """
        Initialize the `RandomClearLabel` module.

        Parameters
        ----------
        prob : Union[float, int, Sampler], optional
            Probability of any label/region being selected for erasure as determined by iid
            Bernoulli trials, by default 0.5.
        exclude_zero : bool, optional
            Optionally exclude zero (uaually background) from the list of potential regions to clear
            (never clear zero labels), by default True.
        seed : int, optional
            A random seed or sampler to control the randomness of label clearing operations. If
            provided, it ensures reproducibility of the clearing process. By default, None.
        """
        super().__init__()
        self.prob = prob
        self.exclude_zero = exclude_zero
        self.seed = seed

    def forward(
        self,
        input_tensor: torch.Tensor,
        label_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClearLabel` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image or tensor to clear.
        label_tensor : torch.Tensor
            Label map corresponding to sampling domain from which to select regions for clearing.

        Returns
        -------
        torch.Tensor
            The modified tensor with specified labels cleared (set to zero). If no labels are
            cleared, the original `input_tensor` is returned unchanged.
        """

        return utils.random_clear_label(
            input_tensor=input_tensor,
            label_tensor=label_tensor,
            prob=self.prob,
            exclude_zero=self.exclude_zero,
            seed=self.seed
        )


class SampleImageFromLabels(BaseTransform):
    """
    A PyTorch module that generates an image from a label map by uniformly sampling a random
    intensity for each label.

    `SampleImageFromLabels` identifies all unique integer labels in the `label_tensor`, and assigns
    each a mean intensity to the labeled region in the corresponding output image (`sampled_image`).
    The mean intensity serves as the mean for a noise distribution modeled by `noise_sampler`. The
    variance of the noise model may be a fixed quantity or sampled from another distribution defined
    by `noise_variance`.
    """

    def __init__(
        self,
        mean_sampler: Sampler = Uniform(0, 1),
        noise_sampler: Sampler = Normal,
        noise_variance: Union[float, int, Sampler] = 0.25,
        *args,
        **kwargs
    ):
        """
        Initialize the `SampleImageFromLabels` module.

        Parameters
        ----------
        mean_sampler : Sampler
            A `Sampler` from which to draw the mean intensity for each region defined by each label
            in the `label_tensor`. By default, `Uniform(0, 1)`
        noise_sampler : Sampler
            A `Sampler` that is used to model the noise within a particular label/region. The mean
            for the sampler is defined by the mean region intensity (sampled from `mean_sampler`).
            By default, `Normal`.
        noise_variance : float, int, or Sampler
            The variance of the noise model. It can be a fixed quantity (int or float), or a sampled
            quantity in the case a `Sampler` is passed. By default, 0.25.
        """
        super().__init__(
            mean_sampler=mean_sampler,
            noise_sampler=noise_sampler,
            noise_variance=noise_variance,
            *args,
            **kwargs
        )

    def transform(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the transformation for the `SampleImageFromLabels` module.

        Parameters
        ----------
        label_tensor : torch.Tensor
            A tensor with batch and channel dimensions containing integer labels defining distinct
            regions.

        Returns
        -------
        torch.Tensor
            A tensor of sampled image intensities with the same shape as `label_tensor`.
        """
        return utils.sample_image_from_labels(
            label_tensor,
            self.mean_sampler,
            self.noise_sampler,
            self.noise_variance
        )


class DrawImage(SampleImageFromLabels):
    """
    @deprecated: Use `SampleImageFromLabels` instead.
    A PyTorch module that generates an image from a label map by uniformly sampling a random
    intensity for each label.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DrawImage is deprecated and will be removed in future versions. "
            "Use SampleImageFromLabels instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class Norm(nn.Module):
    """
    Dynamically constructs a normalization layer based on `norm_type` and `ndim`.

    Supports
    --------
    - 'batch': BatchNorm1d, BatchNorm2d, BatchNorm3d (requires `ndim`)
    - 'instance': InstanceNorm1d, InstanceNorm2d, InstanceNorm3d (requires `ndim`)
    - 'layer': LayerNorm (does not use `ndim`)
    - 'group': GroupNorm (requires `num_groups`)

    The user can also provide a custom `nn.Module` class in `norm_type`.
    """

    # Map normalization types and dimensions for their corresponding PyTorch classes
    NORMALIZATION_MAP = {
        "batch": {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        },
        "instance": {
            1: nn.InstanceNorm1d,
            2: nn.InstanceNorm2d,
            3: nn.InstanceNorm3d,
        },
        "layer": nn.LayerNorm,
        "group": nn.GroupNorm
    }

    def __init__(
        self,
        norm_type: Union[str, Type[nn.Module]],
        ndim: Optional[int] = None,
        num_features: Optional[int] = None,
        num_groups: Optional[int] = None,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        """
        Initialize the `Norm` module.

        Parameters
        ----------
        norm_type : str or nn.Module
            Type of normalization. Must be one of 'batch', 'instance', 'layer',
            'group', or a custom `nn.Module` class.
        ndim : int, optional
            Dimensionality for batch/instance normalization:
            - 1 -> *Norm1d
            - 2 -> *Norm2d
            - 3 -> *Norm3d
            Required for 'batch' or 'instance' norms.
        num_features : int, optional
            Number of input features or channels. Required for 'batch', 'instance',
            'layer', and 'group' normals. For layer norm, this is the size of the
            normalized dimension. For batch and instance norms, this is typically the
            number of channels/features.
        num_groups : int, optional
            Number of groups for GroupNorm. Required for 'group' normalization.
        eps : float, optional
            A value added to the denominator for numerical stability. Default is 1e-5.
        affine : bool, optional
            If True, the layer has learnable affine parameters. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments are passed directly to the normalization class
            constructor. This enables further customization without modifying this class.

        Examples
        --------
        >>> # Dummy input of shape (B, C, H, W)
        >>> x = torch.randn(1, 16, 32, 32)

        ### Using a custom, pre-instantiated normalization layer
        >>> norm_a = nn.InstanceNorm2d(16)
        >>> norm_A = Norm(norm_a)
        >>> norm_A(x)
        ...

        ### Using a custom normalization layer that has not been instantiated
        >>> norm_b = nn.InstanceNorm2d
        >>> norm_B = Norm(norm_b, num_features=16)
        >>> norm_B(x)
        ...

        ### Using the wrapper to define a normalization layer
        >>> norm_C = Norm(norm_type='instance', ndim=2, num_features=16)
        >>> norm_C(x)
        ...
        """
        super().__init__()

        # Norm object has been instantiated with parameters
        if utils.is_instantiated_normalization(norm_type):
            self.norm = norm_type
            return

        # Norm object has been provided but not instantiated
        if isinstance(norm_type, type) and issubclass(norm_type, nn.Module):
            # Assume user provided a custom normalization class directly
            if num_features is None:
                raise ValueError("`num_features` must be specified for custom norms.")
            self.norm = norm_type(num_features=num_features, eps=eps, affine=affine, **kwargs)
            return

        # Handle known norm_types
        if norm_type not in self.NORMALIZATION_MAP:
            raise ValueError(
                f"Invalid norm_type '{norm_type}'. Must be one of "
                f"{list(self.NORMALIZATION_MAP.keys())} or a custom nn.Module subclass."
            )

        # Batch and instance norm require an input dimensionality
        if norm_type in ("batch", "instance"):
            if ndim not in (1, 2, 3):
                raise ValueError(
                    "For 'batch' or 'instance' normalization, ndim must be 1, 2, or 3."
                )
            # They also require the number of features
            if num_features is None:
                raise ValueError("`num_features` must be specified for 'batch' or 'instance' norm.")

            norm_class = self.NORMALIZATION_MAP[norm_type][ndim]
            self.norm = norm_class(
                num_features=num_features, eps=eps, affine=affine, **kwargs
            )

        elif norm_type == "layer":
            if num_features is None:
                raise ValueError(
                    "`num_features` (normalized shape) must be specified for 'layer' norm."
                )
            self.norm = nn.LayerNorm(
                num_features, eps=eps, elementwise_affine=affine, **kwargs
            )

        elif norm_type == "group":
            if num_groups is None:
                raise ValueError("For 'group' normalization, `num_groups` must be specified.")
            if num_features is None:
                raise ValueError("`num_features` must be specified for 'group' norm.")
            self.norm = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine, **kwargs)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the normalization layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to be normalized. The shape depends on the normalization type:
            - For 1D norms: (N, C, L)
            - For 2D norms: (N, C, H, W)
            - For 3D norms: (N, C, D, H, W)
            - For layer/group norms: shape can vary, but typically (N, *)

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        return self.norm(input_tensor)


class Activation(nn.Module):
    """
    Dynamically constructs an activation layer based on the specified type.
    """

    def __init__(
        self,
        activation_type: str,
        inplace: bool = True,
        negative_slope: float = 0.01,
        alpha: float = 1.0
    ):
        """
        Initialize the `Activation` module.

        Parameters
        ----------
        activation_type : str
            Type of activation function. Supported values: 'relu',
            'leaky_relu',
            'elu'.
        inplace : bool, optional
            Whether to perform the operation in-place. Default is True.
        negative_slope : float, optional
            Negative slope for 'leaky_relu'. Default is 0.01.
        alpha : float, optional
            Alpha value for 'elu'. Default is 1.0.
        """
        super(Activation, self).__init__()
        if isinstance(activation_type, torch.nn.Module):
            self.activation = activation_type
        elif activation_type == "relu":
            self.activation = nn.ReLU(inplace=inplace)
        elif activation_type == "leaky_relu":
            self.activation = nn.LeakyReLU(
                negative_slope=negative_slope, inplace=inplace
            )
        elif activation_type == "elu":
            self.activation = nn.ELU(alpha=alpha, inplace=inplace)
        else:
            raise ValueError(
                f"Unsupported activation_type '{activation_type}'. "
                f"Supported types: 'relu', 'leaky_relu', 'elu'."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the activation layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """
        return self.activation(input_tensor)


#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################


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
