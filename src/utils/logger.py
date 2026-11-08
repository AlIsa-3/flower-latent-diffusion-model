import logging


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger("ModelTrainingLogger")
    logger.addHandler(logging.NullHandler())
    return logger
