import torch
import os

import logging

logger = logging.getLogger(__name__)


def set_precision(data_type="float"):
    """This function allows the user to set the default precision. Call before declaring any variables!

    Args:
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".
    """
    if torch.cuda.is_available() and enable_cuda == True:
        if data_type.lower() == "float":
            print("Setting default tensor type to cuda.Float32")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif data_type.lower() == "double":
            print("Setting default tensor type to cuda.Float64")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif:
        if data_type.lower() == "float":
            print("Setting default tensor type to Float32")
            torch.set_default_tensor_type(torch.FloatTensor)
        elif data_type.lower() == "double":
            print("Setting default tensor type to Float64")
            torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        logger.warn(
            "Error enabling CUDA. cuda.is_available() returned False. CPU will be used."
        )
