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

        left_finder = car.processing.LaneLineFinder(mask[:, :mask.shape[1]//2], offset=0)
        right_finder = car.processing.LaneLineFinder(mask[:, (mask.shape[1] // 2):], offset=mask.shape[1] // 2)

        left_lane_rough_sketch = left_finder.get_lane_drawing()
        right_lane_rough_sketch = right_finder.get_lane_drawing()

        left_lane_equation = left_finder.get_lane_equation()
        right_lane_equation = right_finder.get_lane_equation()

        unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
        left_lane_mask = car.processing.get_lane_mask(undistorted_image, left_lane_equation, unwarp_matrix)
        right_lane_mask = car.processing.get_lane_mask(undistorted_image, right_lane_equation, unwarp_matrix)

        image_with_lanes = undistorted_image.copy().astype(np.float32)

        image_with_lanes[left_lane_mask == 1] = (0, 0, 255)
        image_with_lanes[right_lane_mask == 1] = (0, 0, 255)

        images = [undistorted_image, warped, 255 * mask, 255 * left_lane_rough_sketch, 255 * right_lane_rough_sketch,
                  255 * left_lane_mask, image_with_lanes]

        logger.info(vlogging.VisualRecord("Image, processed",
                                          [cv2.resize(
                                              image,
                                              (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5)))
                                           for image in images]))


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    find_lane_lines_in_test_images(logger)


if __name__ == "__main__":

    main()