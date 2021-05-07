import torch

import logging

logger = logging.getLogger(__name__)


def set_precision(data_type="float"):
    """This function allows the user to set the default precision. Call before declaring your variables.

    Args:
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".

    Raises:
        ValueError: If illegal data type.

    """

    if torch.cuda.is_initialized():
        if data_type.lower() == "double":
            logger.warn(
                "Setting default tensor type to cuda.Float64 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif data_type.lower() == "float":
            logger.warn(
                "Setting default tensor type to cuda.Float32 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            logging.error(
                "Illegal data type. Setting default tensor type to cuda.Float32 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if data_type.lower() == "double":
            logger.warn(
                "Setting default tensor type to Float64 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "float":
            logger.warn(
                "Setting default tensor type to Float32 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            logging.error("Illegal data type. Setting default tensor type to Float32.")
            torch.set_default_tensor_type(torch.FloatTensor)