import torch

import logging

logger = logging.getLogger(__name__)


def set_precision(data_type="float"):
    """This function allows the user to set the default precision. Call before declaring your variables.

    Args:
        data_type (string, optional): Data type to use, either "float"/"float32", "double"/"float64", "cfloat"/"complex64", or "cdouble"/"complex128". Defaults to "float".

    """

    if torch.cuda.is_initialized():
        if data_type.lower() == "double" or data_type.lower() == "float64":
            logging.info(
                "Setting default tensor type to cuda.DoubleTensor (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif data_type.lower() == "float" or data_type.lower() == "float32":
            logging.info(
                "Setting default tensor type to cuda.FloatTensor (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif data_type.lower() == "cdouble" or data_type.lower() == "complex128":
            # logging.info(
            logging.error(
                # "Setting default tensor type to cuda.cdouble (i.e. complex128) (CUDA is initialized)."
                "Setting default tensor type to cuda.DoubleTensor (CUDA is initialized). "
                "CUDA does not yet allow for complex tensors."
            )
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif data_type.lower() == "cfloat" or data_type.lower() == "complex64":
            # logging.info(
            logging.error(
                # "Setting default tensor type to cuda.cfloat (i.e. complex64) (CUDA is initialized)."
                "Setting default tensor type to cuda.FloatTensor (CUDA is initialized). "
                "CUDA does not yet allow for complex tensors."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            logging.error(
                data_type,
                "Invalid data type. Only float (float32), double (float64), cfloat (complex64), and cdouble (complex128) are supported. "
                "Setting default tensor type to cuda.FloatTensor (CUDA is initialized). "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if data_type.lower() == "double" or data_type.lower() == "float64":
            logging.info(
                "Setting default tensor type to DoubleTensor (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "float" or data_type.lower() == "float32":
            logging.info(
                "Setting default tensor type to FloatTensor (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        elif data_type.lower() == "cdouble" or data_type.lower() == "complex128":
            # logging.info(
            logging.error(
                # "Setting default tensor type to complex128 (CUDA not initialized)."
                "Setting default tensor type to DoubleTensor (CUDA not initialized). "
                "Torch does not yet allow for complex tensors."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "cfloat" or data_type.lower() == "complex64":
            # logging.info(
            logging.error(
                # "Setting default tensor type to complex64 (CUDA not initialized)."
                "Setting default tensor type to FloatTensor (CUDA not initialized). "
                "Torch does not yet allow for complex tensors."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            logging.error(
                data_type,
                "Invalid data type. Only float (float32), double (float64), cfloat (complex64), and cdouble (complex128) are supported. "
                "Setting default tensor type to FloatTensor. "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.FloatTensor)
