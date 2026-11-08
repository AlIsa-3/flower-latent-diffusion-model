import logging
import pathlib
from datetime import datetime


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger("ModelTrainingLogger")
    logger.addHandler(logging.NullHandler())
    return logger


def setup_logger(
    filename: str = "training",
    log_dir: str | pathlib.Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_filename = log_dir / f"{timestamp}_{filename}.log"

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(level)

    # Prevent messages from being passed to the root logger.
    logger.propagate = False

    # Remove existing handlers to prevent duplicate logging.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Configure file handler.
    file_handler = logging.FileHandler(
        logs_filename,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
