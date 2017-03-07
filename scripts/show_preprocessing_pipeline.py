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
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9
    }

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    for path in paths:

        image = cv2.imread(path)
        undistorted_image = preprocessor.get_undistorted_image(image)

        saturation = preprocessor.get_saturation_mask(undistorted_image)
        x_gradient = preprocessor.get_x_direction_gradient_mask(undistorted_image)

        images = [image, 255 * saturation, 255 * x_gradient]

        target_size = (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5))
        logger.info(vlogging.VisualRecord("Images", [cv2.resize(image, target_size) for image in images]))


if __name__ == "__main__":

    main()