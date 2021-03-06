"""Provides the logger for every program."""


import logging
from datetime import datetime


def get_logger() -> logging.Logger:
    """Gets logger, which logs to logs folder and to the terminal.

    Returns:
        logger (logging.Logger): the logger.
    """
    logger = logging.getLogger('defy')
    logger.setLevel(level=logging.DEBUG)

    filename = (f'logs/{datetime.now()}.log').replace(':', '')
    file_handler = logging.FileHandler(filename)
    stream_handler = logging.StreamHandler()

    format = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    file_handler.setFormatter(format)
    stream_handler.setFormatter(format)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
