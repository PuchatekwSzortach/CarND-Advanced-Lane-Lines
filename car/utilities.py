"""
Sundry utilities
"""

import logging
import os

import cv2


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


def get_image(path):
    """
    Moviepy reads in RGB order, so make sure we do so in OpenCV as well
    """

    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)