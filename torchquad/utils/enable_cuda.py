import torch
import os
from loguru import logger

from .set_precision import set_precision


def enable_cuda(device=0, data_type="float"):
    """This function will set the default device to CUDA if possible. Call before declaring any variables!
    The default precision can be set here initially, or using set_precision later.

    Args:
        device (int, optional): CUDA device to use. Defaults to 0.
        data_type (string, optional): Data type to use, either "float" or "double". Defaults to "float".

    """
    if torch.cuda.is_available():
        os.environ["TORCH_DEVICE"] = "cuda:" + str(device)
        logger.info("__pyTorch VERSION:" + str(torch.version))
        logger.info("__CUDNN VERSION:" + str(torch.backends.cudnn.version()))
        logger.info("__Number of CUDA Devices:" + str(torch.cuda.device_count()))
        logger.info("Active CUDA Device: GPU" + str(torch.cuda.current_device()))
        set_precision(data_type)
    else:
        logger.warning(
            "Error enabling CUDA. cuda.is_available() returned False. CPU will be used."
        )
