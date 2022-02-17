from loguru import logger

from .set_precision import set_precision
from .enable_cuda import enable_cuda


def set_up_backend(backend, data_type=None, torch_enable_cuda=True):
    """Configure a numerical backend for torchquad.

    With the torch backend, this function calls torchquad.enable_cuda unless torch_enable_cuda is False.
    With the tensorflow backend, this function enables tensorflow's numpy behaviour, which is a requirement for torchquad.
    If a data type is passed, set the default floating point precision with torchquad.set_precision.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        data_type ("float32", "float64" or None, optional): Data type which is passed to set_precision. If None, do not call set_precision except if CUDA is enabled for torch. Defaults to None.
        torch_enable_cuda (Bool, optional): If True and backend is "torch", call enable_cuda. Defaults to True.
    """
    if backend == "torch":
        if torch_enable_cuda:
            enable_cuda()
    elif backend == "tensorflow":
        from tensorflow.python.ops.numpy_ops import np_config

        logger.info("Enabling numpy behaviour for Tensorflow")
        # The Tensorflow backend only works with numpy behaviour enabled.
        np_config.enable_numpy_behavior()
    if data_type is not None:
        set_precision(data_type, backend=backend)
