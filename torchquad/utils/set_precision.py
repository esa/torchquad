from loguru import logger


# Precisions for Numpy and Tensorflow if integration_domain is a list
torchquad_default_dtypes = {"numpy": None, "tensorflow": None}


def _set_precision_torch(data_type="float"):
    """This function allows the user to set the default precision for torch.
    Call before declaring your variables.

    Args:
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".

    """
    import torch

    if torch.cuda.is_initialized():
        if data_type.lower() == "double":
            logger.info(
                "Setting Torch's default tensor type to cuda.Float64 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif data_type.lower() == "float":
            logger.info(
                "Setting Torch's default tensor type to cuda.Float32 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            logger.error(
                data_type,
                "Invalid data type. Only float or double are supported. "
                "Setting default tensor type of Torch to cuda.Float32 (CUDA is initialized). "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if data_type.lower() == "double":
            logger.info(
                "Setting Torch's default tensor type to Float64 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "float":
            logger.info(
                "Setting Torch's default tensor type to Float32 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            logger.error(
                data_type,
                "Invalid data type. Only float or double are supported. "
                "Setting default tensor type of Torch to Float32. "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.FloatTensor)


def set_precision(data_type="float", backend="torch"):
    """This function allows the user to set the default precision for floating point numbers for the given numerical backend.
    Call before declaring your variables.
    Numpy and Tensorflow are not supported:
    https://github.com/numpy/numpy/issues/6860
    https://github.com/tensorflow/tensorflow/issues/26033

    Args:
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".
        backend (string, optional): Numerical backend for which the data type is changed. Defaults to "torch".
    """
    if backend == "torch":
        _set_precision_torch(data_type)
    elif data_type not in ["float", "double"]:
        logger.error(
            f'Invalid data type for {backend}: "{data_type}". Only float or double are supported.'
        )
    elif backend == "jax":
        from jax.config import config

        config.update("jax_enable_x64", data_type == "double")
        logger.info(f"JAX data type set to {data_type}")
    elif backend in ["numpy", "tensorflow"]:
        torchquad_default_dtypes[backend] = {"float": "float32", "double": "float64"}[
            data_type
        ]
        logger.info(
            f"Default dtype config for backend {backend} set to {torchquad_default_dtypes[backend]}"
        )
    else:
        logger.error(f"Changing the data type is not supported for backend {backend}")
