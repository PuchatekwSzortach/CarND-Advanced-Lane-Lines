"""
Sundry utilities
"""

import logging
import os


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("simulation")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
