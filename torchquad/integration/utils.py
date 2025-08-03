"""
Utility functions for the integrator implementations including extensions for
autoray, which are registered when importing this file
"""

import sys
from pathlib import Path

# Change the path to import from the parent folder.
# A relative import currently does not work when executing the tests.
sys.path.append(str(Path(__file__).absolute().parent.parent))

from autoray import numpy as anp
from autoray import infer_backend, register_function
from functools import partial
from loguru import logger
import warnings

# from ..utils.set_precision import _get_precision
from utils.set_precision import _get_precision
from utils.set_up_backend import _get_default_backend


def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs.
    Args:
        start (backend tensor): Start point (inclusive).
        stop (backend tensor): End point (inclusive).
        N (int): Number of points.
        requires_grad (bool): Indicates if output should be recorded for backpropagation in Torch.
    Returns:
        backend tensor: Equally spaced 1D grid
    """
    # The requires_grad case is only needed for Torch.
    if requires_grad:
        # Create 0 to 1 spaced grid using the same device and dtype as start
        backend = infer_backend(start)
        if backend == "torch":
            zero = start.new_zeros(())  # scalar tensor
            one = start.new_ones(())  # scalar tensor
        else:
            zero = anp.array(0.0, like=start)
            one = anp.array(1.0, like=start)

        grid = anp.linspace(zero, one, N, dtype=start.dtype)

        # Scale to desired range, thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        if infer_backend(start) == "tensorflow":
            # Tensorflow determines the dtype automatically and doesn't support
            # the dtype argument here
            return anp.linspace(start, stop, N)
        return anp.linspace(start, stop, N, dtype=start.dtype)


def _add_at_indices(target, indices, source, is_sorted=False):
    """
    Add source[i] to target at target[indices[i]] for each index i in-place.
    For example, with targets=[0,0,0] indices=[2,1,1,2] and source=[a,b,c,d],
    targets will be changed to [0,b+c,a+d].
    This function supports only numpy and torch.

    Args:
        target (backend tensor): Tensor to which the source values are added
        indices (int backend tensor): Indices into target for each value in source
        source (backend tensor): Values which are added to target
        is_sorted (bool, optional): Set this to True if indices is monotonically increasing to skip a redundant sorting step with the numpy backend. Defaults to False.
    """
    backend = infer_backend(target)
    if backend == "torch":
        target.scatter_add_(dim=0, index=indices, src=source)
    elif backend == "numpy":
        # Use indicator matrices to reduce the Python interpreter overhead
        # Based on VegasFlow's consume_array_into_indices function
        # https://github.com/N3PDF/vegasflow/blob/21209c928d07c00ae4f789d03b83e518621f174a/src/vegasflow/utils.py#L16
        if not is_sorted:
            # Sort the indices and corresponding source array
            sort_permutation = anp.argsort(indices)
            indices = indices[sort_permutation]
            source = source[sort_permutation]
        # Maximum number of columns for the indicator matrices.
        # A higher number leads to more redundant comparisons and higher memory
        # usage but reduces the Python interpreter overhead.
        max_indicator_width = 500
        zero = anp.array(0.0, dtype=target.dtype, like=backend)
        num_indices = indices.shape[0]
        for i1 in range(0, num_indices, max_indicator_width):
            # Create an indicator matrix for source indices in {i1, i1+1, …, i2-1}
            # and corresponding target array indices in {t1, t1+1, …, t2-1}.
            # All other target array indices are irrelevant: because the indices
            # array is sorted, all values in indices[i1:i2] are bound by t1 and t2.
            i2 = min(i1 + max_indicator_width, num_indices)
            t1, t2 = indices[i1], indices[i2 - 1] + 1
            target_indices = anp.arange(t1, t2, dtype=indices.dtype, like=backend)
            indicator = anp.equal(indices[i1:i2], target_indices.reshape([t2 - t1, 1]))
            # Create a matrix which is zero everywhere except at entries where
            # the corresponding value from source should be added to the
            # corresponding entry in target, sum these source values, and add
            # the resulting vector to target
            target[t1:t2] += anp.sum(anp.where(indicator, source[i1:i2], zero), axis=1)
    else:
        raise NotImplementedError(f"Unsupported numerical backend: {backend}")


def _setup_integration_domain(dim, integration_domain, backend):
    """Sets up the integration domain if unspecified by the user.
    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list or backend tensor or None): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It can also determine the numerical backend.
        backend (string or None): Numerical backend. If set to None, use integration_domain's backend if it is a tensor and otherwise use the backend from the latest call to set_up_backend or "torch" for backwards compatibility.
    Returns:
        backend tensor: Integration domain.
    """
    logger.debug("Setting up integration domain.")

    # If no integration_domain is specified, create [-1,1]^d bounds
    if integration_domain is None:
        integration_domain = [[-1.0, 1.0]] * dim

    # Give an explicitly set backend argument higher precedence than
    # integration_domain's backend.
    # If the backend argument is not None, the dtype of integration_domain is
    # ignored unless its backend and the backend argument are the same.
    domain_arg_backend = infer_backend(integration_domain)
    convert_to_tensor = domain_arg_backend == "builtins"
    if not convert_to_tensor and backend is not None and domain_arg_backend != backend:
        warning_msg = "integration_domain should be a list when the backend argument is set."
        logger.warning(warning_msg)
        warnings.warn(warning_msg, RuntimeWarning)
        convert_to_tensor = True

    # Convert integration_domain to a tensor if needed
    if convert_to_tensor:
        # Cast all integration domain values to Python3 float because
        # some numerical backends create a tensor based on the Python3 types
        integration_domain = [[float(b) for b in bounds] for bounds in integration_domain]
        if backend is None:
            # Get a globally default backend
            backend = _get_default_backend()
        dtype_arg = _get_precision(backend)
        if backend == "tensorflow":
            import tensorflow as tf

            dtype_arg = dtype_arg or tf.keras.backend.floatx()

        integration_domain = anp.array(integration_domain, like=backend, dtype=dtype_arg)

    if integration_domain.shape != (dim, 2):
        raise ValueError(
            "The integration domain has an unexpected shape. "
            f"Expected {(dim, 2)}, got {integration_domain.shape}"
        )

    return integration_domain


def _check_integration_domain(integration_domain):
    """
    Check if the integration domain has a valid shape and determine the dimension.

    Args:
        integration_domain (list or backend tensor): Integration domain, e.g. [[-1,1],[0,1]].
    Returns:
        int: Dimension represented by the domain
    """
    if infer_backend(integration_domain) == "builtins":
        dim = len(integration_domain)
        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

        for bounds in integration_domain:
            if len(bounds) != 2:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
            if anp.any(bounds[0] > bounds[1]):
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
        return dim
    else:
        if len(integration_domain.shape) != 2:
            raise ValueError("The integration_domain tensor has an invalid shape")
        dim, num_bounds = integration_domain.shape
        if dim < 1:
            raise ValueError("integration_domain.shape[0] needs to be 1 or larger.")
        if num_bounds != 2:
            raise ValueError("integration_domain must have 2 values per boundary")
        # The boundary values check does not work if the code is JIT compiled
        # with JAX or TensorFlow.
        if _is_compiling(integration_domain):
            return dim
        if anp.min(integration_domain[:, 1] - integration_domain[:, 0]) < 0.0:
            raise ValueError("integration_domain has invalid boundary values")
        return dim


# Register anp.repeat for torch
@partial(register_function, "torch", "repeat")
def _torch_repeat(a, repeats, axis=None):
    import torch

    # torch.repeat_interleave corresponds to np.repeat and should not be
    # confused with torch.Tensor.repeat.
    return torch.repeat_interleave(a, repeats, dim=axis)


@partial(register_function, "torch", "expand_dims")
def _torch_expand_dims(a, axis):
    """torch is missing `expand_dims` which appears to exist on all other libraries used.

    Args:
        a (torch.Tensor): Tensor to be expanded along axis
        axis (int): the axis along which to expand the dimensions

    Returns:
        torch.Tensor: a Tensor with an extra dimension.
    """
    import torch

    return torch.unsqueeze(a, axis)


def expand_func_values_and_squeeze_integral(f):
    """This decorator ensures that the trailing dimension of integrands is indeed the integrand dimension.
    This is pertinent in the 1d case when the sampled values are often of shape `(N,)`.  Then, to maintain backward
    consistency, we squeeze the result in the 1d case so it does not have any trailing dimensions.

    Args:
        f (Callable): the wrapped function
    """

    def wrap(*args, **kwargs):
        # Extract function_values from either positional or keyword arguments
        if len(args) > 1:
            function_values = args[1]
        elif "function_values" in kwargs:
            function_values = kwargs["function_values"]
        else:
            raise ValueError(
                "function_values argument not found in either positional or keyword arguments. "
                "Please provide function_values as the second positional argument or as a keyword argument."
            )

        # i.e we only have one dimension, or the second dimension (that of the integrand) is 1
        is_1d = len(function_values.shape) == 1 or (
            len(function_values.shape) == 2 and function_values.shape[1] == 1
        )

        if is_1d:
            warnings.warn(
                "DEPRECATION WARNING: In future versions of torchquad, an array-like object will be returned."
            )
            if len(args) > 1:
                # Modify positional arguments
                args = (args[0], anp.expand_dims(function_values, axis=1), *args[2:])
            else:
                # Modify keyword arguments
                kwargs["function_values"] = anp.expand_dims(function_values, axis=1)

            result = f(*args, **kwargs)
            return anp.squeeze(result)

        return f(*args, **kwargs)

    return wrap


def _is_compiling(x):
    """
    Check if code is currently being compiled with PyTorch, JAX or TensorFlow

    Args:
        x (backend tensor): A tensor currently used for computations
    Returns:
        bool: True if code is currently being compiled, False otherwise
    """
    backend = infer_backend(x)
    if backend == "jax":
        return any(nam in type(x).__name__ for nam in ["Jaxpr", "JVPTracer"])
    if backend == "torch":
        import torch

        if hasattr(torch.jit, "is_tracing"):
            # We ignore torch.jit.is_scripting() since we do not support
            # compilation to TorchScript
            return torch.jit.is_tracing()
        # torch.jit.is_tracing() is unavailable below PyTorch version 1.11.0
        return type(x.shape[0]).__name__ == "Tensor"
    if backend == "tensorflow":
        import tensorflow as tf

        if hasattr(tf, "is_symbolic_tensor"):
            return tf.is_symbolic_tensor(x)
        # tf.is_symbolic_tensor() is unavailable below TensorFlow version 2.13.0
        return type(x).__name__ == "Tensor"
    return False


def _torch_trace_without_warnings(*args, **kwargs):
    """Execute `torch.jit.trace` on the passed arguments and hide tracer warnings

    PyTorch can show warnings about traces being potentially incorrect because
    the Python3 control flow is not completely recorded.
    This function can be used to hide the warnings in situations where they are
    false positives.
    """
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        return torch.jit.trace(*args, **kwargs)
