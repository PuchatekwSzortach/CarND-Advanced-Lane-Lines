"""
Script for finding lane lines. Adapted for my car
"""

import glob
import os
import time

import cv2
import vlogging
import numpy as np
import moviepy.editor

import car.utilities
import car.config
import car.processing


def find_lane_lines_in_test_images(logger):

    paths = glob.glob(os.path.join(car.config.my_car_test_images_directory, "*.jpeg"))

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    image_shape = (480, 640)
    source = car.processing.get_perspective_transformation_source_coordinates_my_car(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates_my_car(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)

    preprocessor = car.processing.MyCarImagePreprocessor(parameters, warp_matrix)

    for path in paths:

        image = car.utilities.get_image(path)

        warped = preprocessor.get_warped_image(image)
        mask = preprocessor.get_preprocessed_image(image)

        left_finder = car.processing.LaneLineFinder(mask[:, :mask.shape[1] // 2], offset=0)
        right_finder = car.processing.LaneLineFinder(mask[:, (mask.shape[1] // 2):], offset=mask.shape[1] // 2)

        left_search_image = left_finder.get_lane_search_image_without_prior_knowledge()
        right_search_image = right_finder.get_lane_search_image_without_prior_knowledge()

        search_image = np.zeros_like(image)
        search_image[:, :mask.shape[1] // 2] = left_search_image
        search_image[:, mask.shape[1] // 2:] = right_search_image

        image_with_lanes = image.copy().astype(np.float32)

        try:

            left_lane_equation = left_finder.get_lane_equation()
            right_lane_equation = right_finder.get_lane_equation()

            image_with_lanes = car.processing.draw_lane(
                image_with_lanes, left_lane_equation, right_lane_equation, unwarp_matrix)

        except car.processing.LaneSearchError:

            pass

        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), 255 * mask, search_image,
                  cv2.cvtColor(image_with_lanes, cv2.COLOR_BGR2RGB)]

        logger.info(vlogging.VisualRecord(
            "Image, warped, mask, search, lanes", images, fmt='jpg'))


def find_lane_lines_in_videos_smooth(logger):

    path = "../../data/advanced_lanes_detection/my_car.mp4"
    output_path = "../../data/advanced_lanes_detection/my_car_processed.mp4"

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    image_shape = (480, 640)
    source = car.processing.get_perspective_transformation_source_coordinates_my_car(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates_my_car(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)

    preprocessor = car.processing.MyCarImagePreprocessor(parameters, warp_matrix)

    video_processor = car.processing.MyCarSmoothVideoProcessor(preprocessor, source, destination)

    clip = moviepy.editor.VideoFileClip(path)
    processed_clip = clip.fl_image(video_processor.get_image_with_lanes)
    processed_clip.write_videofile(output_path, fps=12, audio=False)


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    # find_lane_lines_in_test_images(logger)
    find_lane_lines_in_videos_smooth(logger)
    #
    # # get_additional_test_frames(logger)


if __name__ == "__main__":

    main()