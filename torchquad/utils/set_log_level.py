"""Log-level configuration for torchquad.

Follows loguru's best practices for library logging (issue #184):

- torchquad's own records are disabled by default at import (see ``__init__.py``),
  so importing the library never adds output to a host application's logging.
- ``set_log_level()`` enables torchquad's records and adds a handler for them,
  tracking its id so repeated calls remove only torchquad's own handlers and
  never touch handlers the host application registered.
- The library never calls a bare ``logger.remove()``. Users can silence torchquad
  again with ``logger.disable("torchquad")`` and re-enable it with
  ``logger.enable("torchquad")``.
"""

import sys

from loguru import logger

# Ids of the handlers torchquad has added. Tracking them lets set_log_level
# remove only its own handlers when reconfiguring and leave the host
# application's handlers untouched.
_torchquad_handler_ids = []


def set_log_level(log_level):
    """Set the log level for torchquad's own log records.

    torchquad is silent until this is called. Setting the TORCHQUAD_LOG_LEVEL
    environment variable before importing torchquad calls it automatically with
    that level.

    Args:
        log_level (str): The log level to set. Options are 'TRACE', 'DEBUG',
            'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    # torchquad's records are disabled by default (see __init__.py); enable them
    # once the user opts into logging by setting a level.
    logger.enable("torchquad")

    # Remove only torchquad's previously added handlers, never the host
    # application's. An id can go stale if the host reset loguru (e.g. a bare
    # logger.remove()), so tolerate its absence instead of crashing.
    for handler_id in _torchquad_handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            pass
    _torchquad_handler_ids.clear()

    handler_id = logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green>|TQ-<blue>{level}</blue>| <level>{message}</level>",
        filter="torchquad",
    )
    _torchquad_handler_ids.append(handler_id)
    logger.debug(f"Setting LogLevel to {log_level}")
