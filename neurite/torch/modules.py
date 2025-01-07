"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.
"""
__all__ = [
    "Norm",
    "Activation",
    "Conv",
    "ConvBlock",
    "TransposedConv",
    "Pool",
    "EncoderBlock",
    "DecoderBlock"
]

from typing import Union, Type, Optional
import torch
from torch import nn
from . import utils


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
        norm_type: Union[str, Type[nn.Module], None],
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
        activation_type: Union[str, Type[nn.Module], None] = None,
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
        if activation_type is None:
            self.activation = None
        elif isinstance(activation_type, torch.nn.Module):
            self.activation = activation_type
        elif isinstance(activation_type, type) and issubclass(activation_type, nn.Module):
            self.activation = activation_type()
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
        if self.activation is None:
            return nn.Identity()(input_tensor)
        else:
            return self.activation(input_tensor)


class Conv(nn.Module):
    """
    Dynamically constructs an n-dimensional convolutional layer (Conv1d, Conv2d, or Conv3d) based on
    the input dimensionality `ndim`.

    This module enables flexible construction of convolutional layers for 1D, 2D, or 3D data, by
    internally selecting the appropriate PyTorch convolution operation (`torch.nn.Conv1d`,
    `torch.nn.Conv2d`, or `torch.nn.Conv3d`).
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize the `Conv` module.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).

            - 1: Uses `torch.nn.Conv1d` and expects input tensors of shape
              `(N, C, L)`, where `N` is the batch size, `C` is the number of
              input channels, and `L` is the length of the input sequence.

            - 2: Uses `torch.nn.Conv2d` and expects input tensors of shape
              `(N, C, H, W)`, where `H` and `W` are the spatial dimensions of the input image or
              feature map.

            - 3: Uses `torch.nn.Conv3d` and expects input tensors of shape
              `(N, C, D, H, W)`, where `D`, `H`, and `W` are the spatial dimensions of the input
              image or feature map.

        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple
            Size of the convolving kernel.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels.
            Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        """
        super(Conv, self).__init__()

        # Mapping of spatial dimensions for convolutions
        conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

        # Determine if `ndim` is valid
        if ndim not in conv_dim_map:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Dynamically retreive nn.convXd
        conv_cls_name = f"Conv{conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Init the dynamically retreived conv class with necessary input arguments
        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Convolved output tensor.
        """
        return self.conv(input_tensor)


class ConvBlock(nn.Sequential):
    """
    Convolutional Block comprising a convolutional layer, and optionally, an activation function and
    normalization.

    The default sequence of operations in this block is:
        1. **Convolution**: Applies a 2D convolution over the input.
        2. **Normalization**: Normalizes the output of the convolution to stabilize and accelerate
        training.
        3. **Activation Function**: Introduces non-linearity to the model.

    Attributes
    ----------
    conv : nn.Conv2d
        The convolutional layer.
    batch_norm : nn.BatchNorm2d
        The batch normalization layer.
    activation : nn.Module
        The activation function.

    Examples
    --------
    >>> import torch.nn as nn
    >>> conv_block = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU()
        )
    >>> input_tensor = torch.randn(16, 64, 32, 32)
    >>> output = conv_block(input_tensor)
    >>> print(output.shape)
    torch.Size([16, 128, 32, 32])
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = None,
        order: str = 'nca'
    ):
        """
        Initialize the `ConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple, optional
            Size of the convolving kernel. Default is 3.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Every `dilation`-th element is used. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels. Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        norm : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Norm` module: Instantiated or uninstantiated `Norm` layer.
                e.g. nn.InstanceNorm3d(16) or nn.InstanceNorm3d
            - `None`: No normalization is applied. Default is `None`.

        activation : str, nn.Module, or None, optional
            Defines the activation layer. Can be one of:
            - A string: Supported options are 'relu', 'leaky_relu', or 'elu'.
            - A `nn.Module`: Instantiated or uninstantiated activation module.
                e.g. nn.Sigmoid(), nn.Sigmoid
            - `None`: No activation is applied. Default is `None`.

        order : str, optional
            The order of operations in the block. Default is 'nca'
            (normalization -> convolution -> activation).
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation

        Examples
        --------
        ### Basic usage with default options:
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm="batch",
                activation="relu"
            )
        >>> input_tensor = torch.randn(1, 16, 64, 64)
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Using a pre-instantiated `Norm` and `Activation` module:
        >>> norm_layer = nn.BatchNorm2d(32)
        >>> activation_layer = nn.ReLU()
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm=norm_layer,
                activation=activation_layer
            )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Omitting normalization or activation:
        >>> conv_block = ConvBlock(
            ndim=2,
            in_channels=16,
            out_channels=32,
            norm=None,
            activation=None
        )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])
        """
        self.order = order
        # Tracker to determine the logic for the first convolution as opposed to later ones.
        first_conv = True
        num_features = in_channels

        if order is None:
            order = 'nca'
        # Break order into list of letters (operations)
        self.order = list(order)

        # Validate the operations
        valid_operations = ['c', 'n', 'a']
        if not set(order).issubset(valid_operations):
            raise ValueError(
                f"Invalid order. Must be a subset of {valid_operations}."
            )

        # Init layers container
        layers = []
        # Collect layers in the appropriate order
        for operation in self.order:
            if operation == 'c':
                # Make convolution
                layers.append(
                    Conv(
                        ndim,
                        num_features,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias
                    )
                )
                if first_conv:
                    first_conv = False
                    num_features = out_channels

            elif operation == 'n' and norm is not None:
                # Make normalization
                layers.append(Norm(norm, ndim, num_features))

            elif operation == 'a' and activation is not None:
                # Make activation
                layers.append(Activation(activation))

        # Create the Sequential container
        super(ConvBlock, self).__init__(*layers)


class TransposedConv(nn.Module):
    """
    Dynamically constructs a transposed convolutional layer (ConvTranspose1d,
    ConvTranspose2d, or ConvTranspose3d) based on the input dimensionality
    `ndim`.

    This module enables flexible definition of transposed convolutional layers
    for 1D, 2D, or 3D data, by internally selecting the appropriate PyTorch
    transposed convolution operation (`torch.nn.ConvTranspose1d`,
    `torch.nn.ConvTranspose2d`, or `torch.nn.ConvTranspose3d`).
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize the `TransposedConv` module.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
            - 1: Uses `torch.nn.ConvTranspose1d` and expects input tensors of shape `(N, C, L)`,
            where `N` is the batch size, `C` is the number of input channels, and `L` is the length
            of the input sequence.

            - 2: Uses `torch.nn.ConvTranspose2d` and expects input tensors of shape `(N, C, H, W)`,
            where `H` and `W` are the spatial dimensions of the input image or feature map.

            - 3: Uses `torch.nn.ConvTranspose3d` and expects input tensors of shape
            `(N, C, D, H, W)`, where `D`, `H`, and `W` are the spatial dimensions of the input
            image or feature map.

        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple
            Size of the convolving kernel.
        stride : int or tuple, optional
            Stride of the convolution. Default is 2.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        output_padding : int or tuple, optional
            Additional size added to one side of each dimension in the output
            shape. Default is 0.
        dilation : int or tuple, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels.
            Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        """
        super(TransposedConv, self).__init__()

        # Mapping of spatial dimensions for convolutions
        conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

        # Determine if `ndim` is valid
        if ndim not in conv_dim_map:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Dynamically retreive nn.convXd
        conv_cls_name = f"ConvTranspose{conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Construct the transposed convolution
        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transposed convolutional layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after transposed convolution.
        """
        return self.conv(input_tensor)


class Pool(nn.Module):
    """
    A pooling layer that dynamically constructs a pooling operation based
    on the dimensionality and pooling mode specified.

    Attributes
    ----------
    pool : nn.Module
        The pooling operation to apply. It is one of `MaxPool`, `AvgPool`, 
        or `LPPool` for 1D, 2D, or 3D inputs.
    """

    def __init__(self, ndim: int, pool_mode: str = 'max', kernel_size=2):
        """
        Initialize the `Pool` module.

        Parameters
        ----------
        ndim : int
            The dimensionality of the pooling operation. Must be 1, 2, or 3.
        pool_mode : str, optional
            The pooling mode to use. Options are 'max' for max pooling, 
            'avg' for average pooling, and 'lp' for LP pooling. Default is
            'max'.
        kernel_size : int or tuple, optional
            The size of the pooling kernel. Default is 2.
        """
        super(Pool, self).__init__()

        # Mapping of pooling operations
        pool_map = {
            'max': 'MaxPool', 'avg': 'AvgPool', 'lp': 'LPPool'
            }

        # Determine if pooling operation is supported
        if pool_mode not in pool_map:
            raise ValueError(f"Unsupported pool_mode={pool_mode}. Must be `max`, `avg`, or `lp`.")

        # Mapping of spatial dimensions for pooling operation
        pool_dim_map = {1: '1d', 2: '2d', 3: '3d'}
        if ndim not in pool_dim_map:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        pool_cls_name = f"{pool_map[pool_mode]}{pool_dim_map[ndim]}"
        pool_cls = getattr(nn, pool_cls_name)
        self.pool = pool_cls(kernel_size=kernel_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be pooled.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the pooling operation.
        """
        return self.pool(input_tensor)


class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a convolutional block followed by a pooling layer.

    Attributes
    ----------
    conv_block : ConvBlock
        The convolutional block applying a series of convolutions, normalization, and activation.
    pool : Pool
        The pooling layer to downsample the feature maps.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        pool_mode: str = "max",
        pool_kernel_size: int = 2,
        order='nca',
    ):
        """
        Initialize the `EncoderBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolving kernel. Default is 3.
        stride : int, optional
            Stride of the convolution. Default is 1.
        padding : int, optional
            Padding added to all sides of the input. Default is 1.
        norm : str, nn.Module, or None, optional
            Normalization type. Default is 'batch'.
        activation : str, nn.Module, or None, optional
            Activation type. Default is 'relu'.
        pool_mode : str, optional
            Pooling mode ('max' or 'avg'). Default is 'max'.
        pool_kernel_size : int, optional
            Kernel size for pooling. Default is 2.
        order : str, optional
            The order of operations in the block. Default is 'nca' (normalization -> convolution ->
            activation). Each character in the string can be specified an arbitrary number of times
            in any order. Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        """
        super().__init__()
        self.conv_block = ConvBlock(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
            order=order,
        )
        self.pool = Pool(ndim=ndim, pool_mode=pool_mode, kernel_size=pool_kernel_size)

    def forward(self, input_tensor: torch.Tensor, return_residual: bool = False) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor after applying convolution and pooling.
        """
        if return_residual:
            conv_resultant = self.conv_block(input_tensor)
            return self.pool(conv_resultant), conv_resultant
        else:
            return self.pool(self.conv_block(input_tensor))


class DecoderBlock(nn.Module):
    """
    Decoder block consisting of a transposed convolution followed by a convolutional block.

    Attributes
    ----------
    upsample : TransposedConv
        The transposed convolutional layer to upsample the feature maps.
    conv_block : ConvBlock
        The convolutional block applying a series of convolutions, normalization, and activation.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        upsample_kernel_size: int = 4,
        upsample_stride: int = 2,
        upsample_padding: int = 1,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        order: str = 'nca'
    ):
        """
        Initialize the `DecoderBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolving kernel. Default is 3.
        stride : int, optional
            Stride of the convolution. Default is 1.
        padding : int, optional
            Padding added to all sides of the input. Default is 1.
        upsample_kernel_size : int, optional
            Kernel size for the transposed convolution. Default is 4.
        upsample_stride : int, optional
            Stride for the transposed convolution. Default is 2.
        upsample_padding : int, optional
            Padding for the transposed convolution. Default is 1.
        norm : str, nn.Module, or None, optional
            Normalization type. Default is 'batch'.
        activation : str, nn.Module, or None, optional
            Activation type. Default is 'relu'.
        order : str, optional
            The order of operations in the block. Default is 'nca' (normalization -> convolution ->
            activation). Each character in the string can be specified an arbitrary number of times
            in any order. Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        """
        super().__init__()
        self.upsample = TransposedConv(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            padding=upsample_padding,
        )
        self.conv_block = ConvBlock(
            ndim=ndim,
            in_channels=in_channels + in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
            order=order
        )

    def forward(self, input_tensor: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Upsampled tensor after applying transposed convolution and further convolutions.
        """
        if isinstance(residual, torch.Tensor):
            features = self.upsample(input_tensor)
            features = torch.cat([features, residual], dim=1)
            return self.conv_block(features)
        else:
            return self.conv_block(self.upsample(input_tensor))
