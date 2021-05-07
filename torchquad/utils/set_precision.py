import torch
import os

import logging

logger = logging.getLogger(__name__)


def set_precision(data_type="float"):
    """This function allows the user to set the default precision. Call before declaring any variables!
    NB: Remember to initialize CUDA first if GPU compatibility is desired.

    Args:
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".
    """

    if torch.cuda.is_initialized():
        if data_type.lower() == "double":
            print("CUDA is initialized. Setting default tensor type to cuda.Float64")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            print("CUDA is initialized. Setting default tensor type to cuda.Float32")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if data_type.lower() == "double":
            print("Setting default tensor type to Float64")
            torch.set_default_tensor_type(torch.DoubleTensor)
        else:
            print("Setting default tensor type to Float32")
            torch.set_default_tensor_type(torch.FloatTensor)
