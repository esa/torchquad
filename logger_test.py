"""This test is to test that logging works properly when importing torchquad."""

import unittest
from contextlib import contextmanager
from loguru import logger

@contextmanager
def capture_logs(level = "INFO", format="{message}"):
    """"Capture loguru-based logs."""
    output = []
    handler_id = logger.add(output.append, level = level, format = format)

    yield output
    logger.remove(handler_id)

class TestLogger(unittest.TestCase):

    def test_logging_with_tq(self):
        import torchquad
        with capture_logs() as cap_log:
            logger.info("This message should print.")
        self.assertEqual(cap_log,["This message should print.\n"])

