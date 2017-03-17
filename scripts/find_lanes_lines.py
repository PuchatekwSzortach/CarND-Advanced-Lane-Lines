"""
Script for finding lane lines
"""

import glob
import os

import cv2
import vlogging
import numpy as np
import moviepy.editor

import car.utilities
import car.config
import car.processing


def find_lane_lines_in_test_images(logger):

    # paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))
    paths = glob.glob(os.path.join(car.config.additional_test_images_directory, "*.jpg"))

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
        destination = car.processing.get_perspective_transformation_destination_coordinates(image.shape)

        image_with_warp_mask = undistorted_image.copy()
        cv2.polylines(image_with_warp_mask, np.int32([source]), isClosed=True, color=(0, 0, 255), thickness=4)

        warp_matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(undistorted_image, warp_matrix, (image.shape[1], image.shape[0]))

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

        images = [image_with_warp_mask, warped, 255 * mask, 255 * left_lane_rough_sketch, 255 * right_lane_rough_sketch, image_with_lanes]

        logger.info(vlogging.VisualRecord("Image, warped, mask, left sketch, right sketch",
                                          [cv2.resize(
                                              image,
                                              (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5)))
                                           for image in images]))


def find_lane_lines_in_videos_simple():

    paths = ["./project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]

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

    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)

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

    # logger = car.utilities.get_logger(car.config.log_path)
    #
    # find_lane_lines_in_test_images(logger)
    # # find_lane_lines_in_videos_simple()
    #
    # # get_additional_test_frames(logger)
    print("FIX BGR 2 RGB ISSUE FIRST!")


if __name__ == "__main__":

    main()