"""
Script for computing camera calibration
"""

import glob
import os
import itertools
import pickle
import random

import cv2
import vlogging
import numpy as np

import car.config
import car.utilities


def compute_camera_calibration(calibration_images_directory, calibration_pickle_path, logger=None):
    """
    Compute calibration parameters and save them to a pickle file
    :param calibration_images_directory: directory with calibration images
    :param calibration_pickle_path: file to save calibration results to
    :param logger: logger
    """

    paths = glob.glob(os.path.join(calibration_images_directory, "*.jpg"))
    images = [cv2.imread(path) for path in paths]

    x_span = 9
    y_span = 6
    pattern_size = (x_span, y_span)

    # Create an iterator that will have elements (0, y, x) and increase first x then y
    board_corners_coordinates = itertools.product([0], range(y_span), range(x_span))

    # Turn iterator into an array and rotate columns order so that each element is (x, y, 0)
    # and elements are increasing first in x, then in y
    board_corners_coordinates = np.fliplr(list(board_corners_coordinates))

    object_space_points = []
    image_space_points = []

    for image in images:

        are_corners_found, corners = cv2.findChessboardCorners(image, patternSize=pattern_size)

        if are_corners_found:

            object_space_points.append(board_corners_coordinates)
            image_space_points.append(corners)

            if logger is not None:

                cv2.drawChessboardCorners(image, pattern_size, corners, are_corners_found)
                logger.info(vlogging.VisualRecord("Image", cv2.pyrDown(image)))

    return_value, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        objectPoints=np.array(object_space_points, dtype=np.float32), imagePoints=np.array(image_space_points),
        imageSize=images[0].shape[1::-1], cameraMatrix=None, distCoeffs=None)

    data = {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion_coefficients
    }

    with open(calibration_pickle_path, "wb") as file:
        pickle.dump(data, file)


def undistort_sample_image(calibration_images_directory, calibration_pickle_path, logger):
    """
    Load camera calibration from a pickle and undistort a sample image from images directory
    :param calibration_images_directory: directory with calibration images
    :param calibration_pickle_path: file with pickle containing camera calibration
    :param logger: logger
    """

    with open(calibration_pickle_path, "rb") as file:

        data = pickle.load(file)

    paths = glob.glob(os.path.join(calibration_images_directory, "*.jpg"))

    path = paths[random.randint(0, len(paths) - 1)]

    image = cv2.imread(path)
    logger.info(vlogging.VisualRecord("Image", cv2.pyrDown(image)))

    undistorted_image = cv2.undistort(image, data['camera_matrix'], data['distortion_coefficients'])
    logger.info(vlogging.VisualRecord("Undistorted image", cv2.pyrDown(undistorted_image)))


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    # compute_camera_calibration(
    #     car.config.calibration_images_directory, car.config.calibration_pickle_path,
    #     logger)

    undistort_sample_image(
        car.config.calibration_images_directory, car.config.calibration_pickle_path,
        logger)


if __name__ == "__main__":

    main()
