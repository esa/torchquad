from loguru import logger
import os


def _get_precision(backend):
    """Get the configured default precision for NumPy or Tensorflow.

    Args:
        backend ("numpy" or "tensorflow"): Numerical backend

    Returns:
        "float32", "float64" or None: Default floating point precision
    """
    return os.environ.get(f"TORCHQUAD_DTYPE_{backend.upper()}", None)


def set_precision(data_type="float32", backend="torch"):
    """This function allows the user to set the default precision for floating point numbers for the given numerical backend.
    Call before declaring your variables.
    NumPy and Tensorflow don't have global dtypes:
    https://github.com/numpy/numpy/issues/6860
    https://github.com/tensorflow/tensorflow/issues/26033
    Therefore, torchquad sets the dtype argument for these two when initialising the integration domain.

    Args:
        data_type (string, optional): Data type to use, either "float32" or "float64". Defaults to "float32".
        backend (string, optional): Numerical backend for which the data type is changed. Defaults to "torch".
    """
    # Backwards-compatibility: allow "float" and "double", optionally with
    # upper-case letters
    data_type = {"float": "float32", "double": "float64"}.get(
        data_type.lower(), data_type
    )
    if data_type not in ["float32", "float64"]:
        logger.error(
            f'Invalid data type "{data_type}". Only float32 and float64 are supported. Setting the data type to float32.'
        )
        data_type = "float32"

    if backend == "torch":
        import torch

        cuda_enabled = torch.cuda.is_initialized()
        tensor_dtype, tensor_dtype_name = {
            ("float32", True): (torch.cuda.FloatTensor, "cuda.Float32"),
            ("float64", True): (torch.cuda.DoubleTensor, "cuda.Float64"),
            ("float32", False): (torch.FloatTensor, "Float32"),
            ("float64", False): (torch.DoubleTensor, "Float64"),
        }[(data_type, cuda_enabled)]
        cuda_enabled_info = (
            "CUDA is initialized" if cuda_enabled else "CUDA not initialized"
        )
        logger.info(
            f"Setting Torch's default tensor type to {tensor_dtype_name} ({cuda_enabled_info})."
        )
        torch.set_default_tensor_type(tensor_dtype)
    elif backend == "jax":
        from jax.config import config

        config.update("jax_enable_x64", data_type == "float64")
        logger.info(f"JAX data type set to {data_type}")
    elif backend in ["numpy", "tensorflow"]:
        os.environ[f"TORCHQUAD_DTYPE_{backend.upper()}"] = data_type
        logger.info(
            f"Default dtype config for backend {backend} set to {_get_precision(backend)}"
        )
    else:
        logger.error(f"Changing the data type is not supported for backend {backend}")
