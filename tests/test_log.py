import logging

import pytest

from src.log import setup_logging


@pytest.fixture
def src_logger():
    """The shared 'src' logger, restored to its original state after the test."""
    logger = logging.getLogger("src")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers.clear()
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_setup_logging_writes_lines_to_the_file(tmp_path, src_logger):
    path = tmp_path / "logs" / "app.log"
    setup_logging(path)
    logging.getLogger("src.example").info("hello from the test")
    src_logger.handlers[0].flush()
    assert "hello from the test" in path.read_text(encoding="utf-8")
    assert "INFO" in path.read_text(encoding="utf-8")


def test_setup_logging_twice_adds_only_one_handler(tmp_path, src_logger):
    setup_logging(tmp_path / "app.log")
    setup_logging(tmp_path / "app.log")
    assert len(src_logger.handlers) == 1


def test_setup_logging_keeps_log_lines_out_of_the_terminal(tmp_path, src_logger):
    setup_logging(tmp_path / "app.log")
    assert src_logger.propagate is False
