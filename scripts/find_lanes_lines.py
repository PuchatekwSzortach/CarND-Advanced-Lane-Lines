"""
Script for finding lane lines
"""

import glob
import os

import cv2
import vlogging
import numpy as np

import car.utilities
import car.config
import car.processing


def find_lane_lines_in_test_images(logger):

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    for path in paths:

        image = cv2.imread(path)

        undistorted_image = preprocessor.get_undistorted_image(image)

        source = car.processing.get_perspective_transformation_source_coordinates(image.shape)

        # cv2.polylines(undistorted_image, np.int32([source]), isClosed=True, color=(0, 0, 255), thickness=4)

        destination = car.processing.get_perspective_transformation_destination_coordinates(image.shape)

        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(undistorted_image, matrix, (image.shape[1], image.shape[0]))

        mask = preprocessor.get_preprocessed_image(warped)

        images = [undistorted_image, warped, 255 * mask]

        target_size = (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5))

        logger.info(vlogging.VisualRecord("Image, processed",
                                          [cv2.resize(image, target_size) for image in images]))


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    find_lane_lines_in_test_images(logger)


if __name__ == "__main__":

    main()