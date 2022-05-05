from loguru import logger

from .set_precision import set_precision


def enable_cuda(data_type="float32"):
    """This function sets torch's default device to CUDA if possible. Call before declaring any variables!
    The default precision can be set here initially, or using set_precision later.

    Args:
        data_type ("float32", "float64" or None, optional): Data type to use. If None, skip the call to set_precision. Defaults to "float32".
    """
    import torch

    if torch.cuda.is_available():
        logger.info("PyTorch VERSION: " + str(torch.__version__))
        logger.info("CUDNN VERSION: " + str(torch.backends.cudnn.version()))
        logger.info("Number of CUDA Devices: " + str(torch.cuda.device_count()))
        logger.info("Active CUDA Device: GPU" + str(torch.cuda.current_device()))
        if data_type is not None:
            set_precision(data_type)
    else:
        logger.warning(
            "Error enabling CUDA. cuda.is_available() returned False. CPU will be used."
        )
