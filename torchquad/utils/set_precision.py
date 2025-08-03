from loguru import logger
import os
import sys


def _get_precision(backend):
    """Get the configured default precision for NumPy or Tensorflow.

    Args:
        backend ("numpy" or "tensorflow"): Numerical backend

    Returns:
        "float32", "float64" or None: Default floating point precision
    """
    return os.environ.get(f"TORCHQUAD_DTYPE_{backend.upper()}", None)


def set_precision(data_type="float32", backend="torch"):
    """Set the default precision for floating-point numbers for the given numerical backend.
    Call before declaring your variables.

    NumPy and doesn't have global dtypes:
    https://github.com/numpy/numpy/issues/6860

    Therefore, torchquad sets the dtype argument for these it when initialising the integration domain.

    Args:
        data_type (str, optional): Data type to use, either "float32" or "float64". Defaults to "float32".
        backend (str, optional): Numerical backend for which the data type is changed. Defaults to "torch".
    """
    # Backwards-compatibility: allow "float" and "double", optionally with
    # upper-case letters
    data_type = {"float": "float32", "double": "float64"}.get(data_type.lower(), data_type)
    if data_type not in ["float32", "float64"]:
        error_msg = f'Invalid data type "{data_type}". Only float32 and float64 are supported. Setting the data type to float32.'
        logger.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        data_type = "float32"

    if backend == "torch":
        import torch

        # Use new PyTorch 2.1+ API if available, fallback to legacy for older versions
        if hasattr(torch, "set_default_dtype"):
            # Modern PyTorch 2.1+ approach
            dtype_map = {"float32": torch.float32, "float64": torch.float64}
            torch.set_default_dtype(dtype_map[data_type])

            # Only set default device to CUDA if CUDA was already initialized
            # (matching the old behavior more closely)
            cuda_enabled = torch.cuda.is_initialized()
            if cuda_enabled:
                torch.set_default_device("cuda")
                logger.info(f"Setting Torch's default dtype to {data_type} and device to CUDA.")
            else:
                logger.info(f"Setting Torch's default dtype to {data_type} (CPU).")
        else:
            # Legacy approach for older PyTorch versions
            cuda_enabled = torch.cuda.is_initialized()
            tensor_dtype, tensor_dtype_name = {
                ("float32", True): (torch.cuda.FloatTensor, "cuda.Float32"),
                ("float64", True): (torch.cuda.DoubleTensor, "cuda.Float64"),
                ("float32", False): (torch.FloatTensor, "Float32"),
                ("float64", False): (torch.DoubleTensor, "Float64"),
            }[(data_type, cuda_enabled)]
            cuda_enabled_info = "CUDA is initialized" if cuda_enabled else "CUDA not initialized"
            logger.info(
                f"Setting Torch's default tensor type to {tensor_dtype_name} ({cuda_enabled_info})."
            )
            torch.set_default_tensor_type(tensor_dtype)
    elif backend == "jax":
        from jax import config

        config.update("jax_enable_x64", data_type == "float64")
        logger.info(f"JAX data type set to {data_type}")
    elif backend == "tensorflow":
        import tensorflow as tf

        # Set TensorFlow global precision
        tf.keras.backend.set_floatx(data_type)
        logger.info(f"TensorFlow default floatx set to {tf.keras.backend.floatx()}")
    elif backend == "numpy":
        # NumPy still lacks global dtype support
        os.environ["TORCHQUAD_DTYPE_NUMPY"] = data_type
        logger.info(f"NumPy default dtype set to {_get_precision('numpy')}")
    else:
        error_msg = f"Changing the data type is not supported for backend {backend}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
