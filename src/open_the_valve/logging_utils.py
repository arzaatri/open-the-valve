import logging
import sys
from pathlib import Path

_LOG_FORMAT = "[%(asctime)s.%(msecs)03d][%(name)s] %(levelname)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, logs_dir: str = "logs") -> None:
    """Configure root logging with a timestamped, module-path-prefixed format.

    Log lines are written both to stdout and to logs/open_the_valve.log.
    Call once at process entrypoint; module loggers should use
    `logging.getLogger(__name__)` so the `%(name)s` field reflects the
    full dotted submodule path.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(Path(logs_dir) / "open_the_valve.log")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [stream_handler, file_handler]
