"""
A script that shows different step of image preprocessing pipeline
"""

import glob
import os

import cv2
import vlogging
import numpy as np

import car.utilities
import car.config
import car.processing


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255]
    }

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    for path in paths:

        image = cv2.imread(path)
        undistorted_image = preprocessor.get_undistorted_image(image)

        saturation = preprocessor.get_saturation_mask(undistorted_image)

        images = [image, undistorted_image, 255 * saturation]

        target_size = (int(image.shape[1] / 3), int(image.shape[0] / 3))
        logger.info(vlogging.VisualRecord("Images", [cv2.resize(image, target_size) for image in images]))


if __name__ == "__main__":

    main()