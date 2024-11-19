"""
Random (samplers) for the neurite project.
"""
__all__ = [
    'ensure_list',
    'Sampler'
]

from typing import Type, Dict, Any, TypeVar, Generator, List, Union, Tuple
import torch

SamplerType = TypeVar('T', bound='Sampler')


def ensure_list(x, size=None, crop=True, **kwargs):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, Generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if size and len(x) < size:
        default = kwargs.get('default', x[-1] if x else None)
        x += [default] * (size - len(x))
    if size and crop:
        x = x[:size]
    return x


class Sampler:
    """
    Base class for random samplers, with a bunch of helpers. This class is for developers of new
    Sampler classes only.
    """

    def __init__(self, **theta):
        """
        Initializes the Sampler with given (arbitrary) parameters `theta`.

        Parameters
        ----------
        **theta : Any
            Arbitrary keyword arguments representing sampler parameters.
        """
        self.theta = theta

    @classmethod
    def make(
        cls: Type[T],
        maker_input: Union['Sampler', Dict[str, Any], Tuple[Any, ...], Any]
    ) -> SamplerType:
        """
        Factory method to create a `Sampler` instance from various input types.

        This method allows the creation of `Sampler` objects from existing `Sampler` instances,
        dictionaries, tuples, or other types by intelligently parsing and forwarding the arguments
        to the appropriate constructor.

        Parameters
        ----------
        maker_input : Union[Sampler, Dict[str, Any], Tuple[Any, ...], Any]
            The input from which to create a `Sampler` instance. It can be:
                - An existing `Sampler` instance: Returned as is.
                - A `dict`: Interpreted as keyword arguments for `Sampler` initialization.
                - A `tuple`: Interpreted as positional arguments for `Sampler` initialization.
                - Any other type: Passed as a single positional argument to `Sampler`
                initialization.

        Returns
        -------
        Sampler
            A `Sampler` instance constructed based on the input type.

        Examples
        --------
        >>> # Case 1: Input is an existing Sampler instance
        >>> sampler1 = Sampler(a=1, b=2)
        >>> sampler2 = Sampler.make(sampler1)
        >>> print(sampler1 is sampler2)
        True

        >>> # Case 2: Input is a dictionary
        >>> params = {'a': 10, 'b': 20}
        >>> sampler3 = Sampler.make(params)
        >>> print(sampler3.theta)
        {'a': 10, 'b': 20}

        >>> # Case 3: Input is a tuple
        >>> params_tuple = (100, 200)
        >>> sampler4 = Sampler.make(params_tuple)
        >>> print(sampler4.theta)
        {'min': 100, 'max': 200}

        >>> # Case 4: Input is another type (e.g., int)
        >>> sampler5 = Sampler.make(300)
        >>> print(sampler5.theta)
        {'min': 0, 'max': 300}
        """
        if isinstance(maker_input, Sampler):
            # If it's a `Sampler`, return as-is
            return maker_input
        elif isinstance(maker_input, dict):
            # Interpret dict as kwargs
            return cls(**maker_input)
        elif isinstance(maker_input, tuple):
            # Interpret as positional args
            if len(maker_input) > 2:
                raise TypeError(
                    f"make expected at most 2 positional arguments in tuple, got {len(maker_input)}"
                )
            return cls(*maker_input)
        else:
            # Pass as single positional arg
            return cls(maker_input)

    def _ensure_same_length(
        self,
        theta: Dict[str, Any],
        nsamples: int = None
    ) -> Dict[str, List[Any]]:
        """
        Ensures that all parameter lists in `theta` have the same length.

        If `nsamples` is specified, it extends or truncates each parameter list in `theta` to match
        `nsamples`. If `nsamples` is not specified, it determines the maximum length among the
        parameter lists and adjusts all lists to that length.

        Parameters
        ----------
        theta : Dict[str, Any]
            A dictionary of parameters where values can be single items or lists/tuples.
        nsamples : int, optional
            The desired length for all parameter lists. If `None`, the maximum existing list length
            is used.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary with all parameter lists adjusted to have the same length.
        """
        # We'll need to make sure `theta` is a dict
        theta = dict(theta)
        if nsamples:
            # We need to bring all params to the known length of `nsamples`
            # Validating `nsamples`
            if not isinstance(nsamples, int) or nsamples <= 0:
                raise ValueError(f"`num_samples` must be a positive integer, got {nsamples}")
            # Make sure all param lists have length `nsamples`
            for param_name, param_value in theta.items():
                theta[param_name] = ensure_list(param_value, nsamples)
        else:
            # We need to determine the max len among all parameter lists (since it's not specified)
            max_length = 0
            for param_value in theta.values():
                if isinstance(param_value, (list, tuple)):
                    max_length = max(max_length, len(param_value))
            if max_length == 0:
                # All parameters are single values; convert them to single-element lists
                for param_name, param_value in theta.items():
                    theta[param_name] = ensure_list(param_value)
            else:
                # Adjust all parameters to the maximum length found
                for param_name, param_value in theta.items():
                    theta[param_name] = ensure_list(param_value, max_length)
        return theta

    @classmethod
    def map(cls, fn: Any, *values: Any, n: int = None) -> Any:
        """
        Applies a function `fn` across multiple lists of values.

        If `n` is specified, each input list is extended or truncated to length `n` before mapping.

        Parameters
        ----------
        fn : Callable
            The function to apply to the mapped values.
        *values : Any
            Multiple lists or single values to be mapped.
        n : int, optional
            The number of samples. If specified, each list in `values` is adjusted to this length.

        Returns
        -------
        Any
            The result of applying `fn` to the mapped values. This could be a list or a single
            value.

        Examples
        --------
        >>> import random
        >>> Sampler.map(random.gauss, [1, 2, 3], [1, 9, 1])
        [0.9751766007463074, 2.4656489254905654, 0.33309233499125224]
        >>> Sampler.map(random.uniform, [0, 1, 2], [10, 20, 100])
        [2.3064614400086403, 7.6623252443254, 99.70018921460807]
        """
        if n:
            values = tuple(ensure_list(value, n) for value in values)
        if isinstance(values[0], list):
            return [fn(*args) for args in zip(*values)]
        else:
            return fn(*values)

    def __getattr__(self, item: str) -> Any:
        """
        Allows dynamic access to parameters stored in `theta`.

        If an attribute is not found through the normal mechanisms, this method is called.
        It attempts to retrieve the attribute from the `theta` dictionary, ensuring all parameter
        lists have the same length.

        Parameters
        ----------
        item : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute from `theta`.

        Examples
        --------
        >>> sampler = Sampler(a=1, b=2)
        >>> sampler.a
        1
        >>> sampler.c
        AttributeError: c
        """
        theta = self.__getattribute__('theta')
        if item in theta:
            theta = self._ensure_same_length(theta)
            return theta[item]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, item: str, value: Any) -> None:
        """
        Allows dynamic setting of parameters stored in `theta`.

        If the attribute exists in `theta`, it updates its value. Otherwise, it sets the attribute
        using the default `__setattr__` behavior.

        Parameters
        ----------
        item : str
            The name of the attribute to set.
        value : Any
            The value to assign to the attribute.
        """
        if item == 'theta':
            return super().__setattr__(item, value)
        if 'theta' in self.__dict__ and item in self.theta:
            self.theta[item] = value
        else:
            return super().__setattr__(item, value)

    def __call__(
        self,
        n: Union[int, List[int]] = None, **backend: Any
    ) -> Union[Any, List[Any], torch.Tensor]:
        """
        Generates samples based on the sampler's parameters.

        This method is intended to be overridden by subclasses to implement specific sampling
        behaviors.

        Parameters
        ----------
        n : Union[int, List[int]], optional
            The number of samples to generate.
            - If `None`, returns a single scalar sample.
            - If an `int`, returns a list of samples.
            - If a `list` of `int`, returns a tensor of samples.

        **backend : Any
            Additional keyword arguments to pass to the backend sampling implementation.

        Returns
        -------
        Union[Any, List[Any], torch.Tensor]
            The generated samples, varying in type based on the `n` parameter.

        Raises
        ------
        NotImplementedError
            Indicates that the method should be implemented by subclasses.
        """

        raise NotImplementedError(
            "The __call__ method must be implemented by subclasses of Sampler."
        )
