"""Log-level configuration for torchquad.

Follows loguru's best practices for library logging (issue #184):

- torchquad is silent by default: its records are disabled at import (see
  ``__init__.py``) and additionally held back by the level gate in
  ``library_logger.py``.
- ``set_log_level()`` is the single switch. It enables torchquad's records and
  sets the lowest severity the library emits.
- torchquad adds no handler of its own and never removes one. Its records go to
  whatever sinks the application has registered with loguru, which in a plain
  interpreter is loguru's default stderr handler.
"""

from loguru import logger

from .library_logger import LEVEL_NUMBERS, set_minimum_level
from .library_logger import logger as library_logger


def set_log_level(log_level):
    """Set the lowest severity of torchquad's own log records.

    torchquad is silent until this is called. Setting the TORCHQUAD_LOG_LEVEL
    environment variable to a non-empty value before importing torchquad calls
    it automatically with that level.

    The level is applied where the records are created, so it holds for every
    loguru handler: records below it are never emitted, and records at or above
    it reach each sink the application has registered exactly once. torchquad
    registers no sink itself. An application that has removed all loguru
    handlers therefore sees nothing until it adds one.

    Args:
        log_level (str): The log level to set. Options are 'TRACE', 'DEBUG',
            'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.

    Raises:
        ValueError: If ``log_level`` is not one of the level names above. The
            environment variable is read at import, so a typo in
            TORCHQUAD_LOG_LEVEL surfaces as a failed ``import torchquad``.
    """
    # Validate before enabling anything, so a typo cannot leave the library
    # half-configured.
    if log_level not in LEVEL_NUMBERS:
        raise ValueError(
            f"Unknown log level {log_level!r}. Expected one of: "
            "TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL."
        )

    # torchquad's records are disabled by default (see __init__.py); enable them
    # once the user opts into logging by setting a level.
    logger.enable("torchquad")
    set_minimum_level(log_level)
    library_logger.debug(f"Setting LogLevel to {log_level}")
