"""
A script that shows different step of image preprocessing pipeline fitted for my footage taken with my car
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


def get_test_frames(logger):

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9
    }

    # preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    path = "../../data/advanced_lanes_detection/my_car.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    times = [10, 60, 150, 180, 240, 300]

    for time in times:

        frame = clip.get_frame(t=time)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
        logger.info(vlogging.VisualRecord(str(time), [frame], fmt="jpg"))


def show_preprocessing_pipeline_for_test_images(logger):

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

    preprocessor = car.processing.MyCarImagePreprocessor(parameters, warp_matrix)

    for path in paths:

        image = car.utilities.get_image(path)

        image_with_warp_mask = image.copy()
        cv2.polylines(image_with_warp_mask, np.int32([source]), isClosed=True, color=(0, 0, 255), thickness=4)

        warped = preprocessor.get_warped_image(image)

        processed = preprocessor.get_preprocessed_image(image)

        images = [
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(image_with_warp_mask, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(warped, cv2.COLOR_RGB2BGR),
            255 * processed
        ]

        logger.info(vlogging.VisualRecord("Image", images, fmt='jpg'))


def show_preprocessing_pipeline_for_test_videos():

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

    clip = moviepy.editor.VideoFileClip(path)

    processed_clip = clip.fl_image(preprocessor.get_preprocessed_image_for_video)

    final_clip = moviepy.editor.clips_array([[clip, processed_clip]])
    final_clip.write_videofile(output_path, fps=12, audio=False)


def main():

    logger = car.utilities.get_logger(car.config.log_path)
    # show_preprocessing_pipeline_for_test_images(logger)

    show_preprocessing_pipeline_for_test_videos()
    # get_test_frames(logger)


if __name__ == "__main__":

    main()