"""
Script for finding lane lines
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

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))
    # paths = glob.glob(os.path.join(car.config.additional_test_images_directory, "*.jpg"))

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    image_shape = (720, 1280)
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters, warp_matrix)

    statistics_computer = car.processing.LaneStatisticsComputer(
        image_shape,
        metres_per_pixel_width=car.config.metres_per_pixel_width,
        metres_per_pixel_height=car.config.metres_per_pixel_height)

    # start = time.time()

    for path in paths:

        image = car.utilities.get_image(path)

        undistorted_image = preprocessor.get_undistorted_image(image)

        warped = preprocessor.get_warped_image(undistorted_image)
        mask = preprocessor.get_preprocessed_image(image)

        left_finder = car.processing.LaneLineFinder(mask[:, :mask.shape[1] // 2], offset=0)
        right_finder = car.processing.LaneLineFinder(mask[:, (mask.shape[1] // 2):], offset=mask.shape[1] // 2)

        left_search_image = left_finder.get_lane_search_image()
        right_search_image = right_finder.get_lane_search_image()

        search_image = np.zeros_like(image)
        search_image[:, :mask.shape[1] // 2] = left_search_image
        search_image[:, mask.shape[1] // 2:] = right_search_image

        left_lane_equation = left_finder.get_lane_equation()
        right_lane_equation = right_finder.get_lane_equation()

        left_lane_mask = car.processing.get_lane_mask(undistorted_image, left_lane_equation, unwarp_matrix)
        right_lane_mask = car.processing.get_lane_mask(undistorted_image, right_lane_equation, unwarp_matrix)

        image_with_lanes = undistorted_image.copy().astype(np.float32)

        image_with_lanes[left_lane_mask == 1] = (0, 255, 0)
        image_with_lanes[right_lane_mask == 1] = (0, 255, 0)

        left_curvature = statistics_computer.get_line_curvature(left_lane_equation)
        right_curvature = statistics_computer.get_line_curvature(right_lane_equation)
        displacement = statistics_computer.get_lane_displacement(left_lane_equation, right_lane_equation)

        cv2.putText(image_with_lanes, "Left lane curvature: {}".format(left_curvature), (100, 80),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        cv2.putText(image_with_lanes, "Right lane curvature: {}".format(right_curvature), (100, 120),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        cv2.putText(image_with_lanes, "Displacement from lane center: {}".format(displacement), (100, 160),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        images = [cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), 255 * mask, search_image,
                  cv2.cvtColor(image_with_lanes, cv2.COLOR_BGR2RGB)]

        logger.info(vlogging.VisualRecord(
            "Image, warped, mask, search, lanes", images))

    # print("Computation time: {}".format(time.time() - start))


def find_lane_lines_in_videos_simple():

    # paths = ["./project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
    paths = ["./project_video.mp4"]

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    image_shape = (720, 1280)
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters, warp_matrix)

    statistics_computer = car.processing.LaneStatisticsComputer(
        image_shape,
        metres_per_pixel_width=car.config.metres_per_pixel_width,
        metres_per_pixel_height=car.config.metres_per_pixel_height)

    video_processor = car.processing.SimpleVideoProcessor(preprocessor, statistics_computer, source, destination)

    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)

        processed_clip = clip.fl_image(video_processor.get_image_with_lanes)

        final_clip = moviepy.editor.clips_array([[clip, processed_clip]])

        output_path = os.path.join(car.config.video_output_directory, os.path.basename(path))
        final_clip.write_videofile(output_path, fps=12, audio=False)


def find_lane_lines_in_videos_smooth():

    # paths = ["./project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
    paths = ["./project_video.mp4"]

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9,
        "y_gradient_thresholds": [10, 50],
        "y_gradient_kernel_size": 9,
        "gradient_magnitude_thresholds": [10, 30],
    }

    image_shape = (720, 1280)
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters, warp_matrix)

    statistics_computer = car.processing.LaneStatisticsComputer(
        image_shape,
        metres_per_pixel_width=car.config.metres_per_pixel_width,
        metres_per_pixel_height=car.config.metres_per_pixel_height)

    video_processor = car.processing.SmoothVideoProcessor(preprocessor, statistics_computer, source, destination)

    for path in paths:

        print("We are only processing a subclip!".upper())
        clip = moviepy.editor.VideoFileClip(path).subclip(20, 27)

        processed_clip = clip.fl_image(video_processor.get_image_with_lanes)

        final_clip = moviepy.editor.clips_array([[clip, processed_clip]])

        output_path = os.path.join(car.config.video_output_directory, os.path.basename(path))
        final_clip.write_videofile(output_path, fps=12, audio=False)


def get_additional_test_frames(logger):

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
    image_shape = [720, 1280]
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)

    video_processor = car.processing.SimpleVideoProcessor(preprocessor, source, destination)

    path = "./project_video.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    for time in [24.5, 41, 41.5]:

        frame = clip.get_frame(t=time)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_frame = video_processor.get_image_with_lanes(frame)
        logger.info(vlogging.VisualRecord(str(time), [frame, processed_frame], fmt="jpg"))


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    # find_lane_lines_in_test_images(logger)
    # find_lane_lines_in_videos_simple()
    find_lane_lines_in_videos_smooth()
    #
    # # get_additional_test_frames(logger)


if __name__ == "__main__":

    main()