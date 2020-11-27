import os
import logging

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"


# Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.base_integrator import BaseIntegrator

__all__ = ["BaseIntegrator"]

# Initialize logger
logger = logging.getLogger(__name__)


def set_log_level(level=logging.WARN):
    """Allow setting global log level for the application

    Args:
        level (logging.level, optional): Level to set, available are (logging.DEBUG,logging.INFO,logging.WARN,logging.ERROR). Defaults to logging.WARN.
    """
    logger.setLevel(level)
    if level == 10:
        logger.info("Log level set to debug")
    elif level == 20:
        logger.info("Log level set to info")
    # we still store it in case we might write some logfile or sth later
    elif level == 30:
        logger.info("Log level set to warn")
    elif level == 40:
        logger.info("Log level set to error")
