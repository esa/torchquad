from loguru import logger
import sys


def set_log_level(log_level: str):
    """Set the log level for the logger
    Args:
        log_level (str): The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green>|TQ-<blue>{level}</blue>| <level>{message}</level>",
    )
    logger.debug(f"Setting LogLevel to {log_level}")
