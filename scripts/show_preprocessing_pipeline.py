"""
A script that shows different step of image preprocessing pipeline
"""

import glob
import os

import cv2
import vlogging

import car.utilities
import car.config


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))

    for path in paths:

        image = cv2.imread(path)

        logger.info(vlogging.VisualRecord("Image", cv2.pyrDown(image)))


if __name__ == "__main__":

    main()