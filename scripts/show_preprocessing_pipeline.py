"""
A script that shows different step of image preprocessing pipeline
"""

import glob
import os

import cv2
import vlogging

import car.utilities
import car.config
import car.processing


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path)

    for path in paths:

        image = cv2.imread(path)
        undistorted_image = preprocessor.get_undistorted_image(image)

        logger.info(vlogging.VisualRecord("Images", [cv2.pyrDown(image), cv2.pyrDown(undistorted_image)]))


if __name__ == "__main__":

    main()