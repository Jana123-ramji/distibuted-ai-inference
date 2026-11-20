"""Shared logging setup.

Both the API and the worker use this so log lines look the same everywhere:

    2026-08-31 09:12:03 INFO  [API] Received inference request
    2026-08-31 09:12:03 INFO  [SCHEDULER] Selected worker-2
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once, writing to stdout (Docker-friendly)."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicated handlers when uvicorn reloads the module.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # uvicorn's access log is noisy while polling the dashboard every 3 s.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
