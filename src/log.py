"""Logging setup: one rotating log file, logs/app.log, shared by the terminal and the Streamlit app."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "app.log"


def setup_logging(path=LOG_PATH):
    """Send everything logged under 'src' to a rotating file (about 1 MB, 3 backups). Safe to call twice."""
    logger = logging.getLogger("src")
    if logger.handlers:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep log lines out of the terminal
