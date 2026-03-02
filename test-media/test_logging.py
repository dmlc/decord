"""Tests for decord logging module."""
import decord
from decord import logging


class TestLogging:
    def test_set_level_quiet(self):
        logging.set_level(logging.QUIET)

    def test_set_level_error(self):
        logging.set_level(logging.ERROR)

    def test_set_level_warning(self):
        logging.set_level(logging.WARNING)

    def test_set_level_info(self):
        logging.set_level(logging.INFO)

    def test_restore_default(self):
        """Restore to ERROR level (the default set in __init__)."""
        logging.set_level(logging.ERROR)
