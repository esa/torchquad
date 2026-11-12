"""The logger every torchquad module uses, with a level gate applied at the source.

loguru decides whether a record is emitted from the calling module's name, and
only its handlers know about levels. Neither is enough for a library:

- ``logger.disable("torchquad")`` does not cover code that a framework re-executes
  under its own module name. TensorFlow's AutoGraph does that to everything inside
  a compiled function, so torchquad's records leaked from the JIT path while
  logging was supposedly off.
- A level set on a handler applies to that handler alone. loguru's default stderr
  handler sits at DEBUG and belongs to the host application, so it printed every
  torchquad record no matter which level the user had asked for.

The gate here drops a record before it reaches loguru unless torchquad's own
level admits it, which holds for every handler and every module name.
"""

import math

from loguru import logger as _loguru_logger

# loguru's standard severities. Hard-coded because the gate must be able to
# compare levels without asking loguru, whose level table a host may extend.
LEVEL_NUMBERS = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class _LevelGate:
    """Holds the lowest severity torchquad currently emits; infinity means silent."""

    minimum_level_number = math.inf


def set_minimum_level(level_name):
    """Let records at or above the given severity through the gate.

    Args:
        level_name (str): One of the keys of :data:`LEVEL_NUMBERS`.
    """
    _LevelGate.minimum_level_number = LEVEL_NUMBERS[level_name]


class _GatedLogger:
    """The subset of loguru's logger interface torchquad uses, behind the level gate."""

    def __init__(self, lazy=False):
        self._lazy = lazy

    def opt(self, lazy=False):
        """Return a logger whose message arguments are callables evaluated on demand.

        Args:
            lazy (bool, optional): Evaluate keyword arguments only if the record is
                emitted. Defaults to False.

        Returns:
            _GatedLogger: A gated logger with the option applied.
        """
        return _GatedLogger(lazy=lazy)

    def _emit(self, level_name, message, arguments, keyword_arguments):
        if LEVEL_NUMBERS[level_name] < _LevelGate.minimum_level_number:
            return
        # depth=2 skips this method and the public one above it, so the record
        # is attributed to the torchquad module that logged it.
        _loguru_logger.opt(depth=2, lazy=self._lazy).log(
            level_name, message, *arguments, **keyword_arguments
        )

    def debug(self, message, *arguments, **keyword_arguments):
        """Log at DEBUG severity; arguments are passed on to loguru for formatting."""
        self._emit("DEBUG", message, arguments, keyword_arguments)

    def info(self, message, *arguments, **keyword_arguments):
        """Log at INFO severity; arguments are passed on to loguru for formatting."""
        self._emit("INFO", message, arguments, keyword_arguments)

    def warning(self, message, *arguments, **keyword_arguments):
        """Log at WARNING severity; arguments are passed on to loguru for formatting."""
        self._emit("WARNING", message, arguments, keyword_arguments)

    def error(self, message, *arguments, **keyword_arguments):
        """Log at ERROR severity; arguments are passed on to loguru for formatting."""
        self._emit("ERROR", message, arguments, keyword_arguments)


logger = _GatedLogger()
