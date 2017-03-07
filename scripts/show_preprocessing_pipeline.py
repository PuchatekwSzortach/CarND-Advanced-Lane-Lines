"""
A script that shows different step of image preprocessing pipeline
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


def show_preprocessing_pipeline_for_test_images(logger):

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

        preprocessed_image = preprocessor.get_preprocessed_image(image)

        images = [image, 255 * saturation, 255 * x_gradient, 255 * preprocessed_image]

        target_size = (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5))
        logger.info(vlogging.VisualRecord("Images", [cv2.resize(image, target_size) for image in images]))


def show_preprocessing_pipeline_for_test_videos():

    paths = ["./project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9
    }

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)
        processed_clip = clip.fl_image(preprocessor.get_preprocessed_image_for_video)

        final_clip = moviepy.editor.clips_array([[clip, processed_clip]])

        output_path = os.path.join(car.config.video_output_directory, os.path.basename(path))
        final_clip.write_videofile(output_path, fps=12, audio=False)


def main():

    # logger = car.utilities.get_logger(car.config.log_path)
    # show_preprocessing_pipeline_for_test_images(logger)

    show_preprocessing_pipeline_for_test_videos()

if __name__ == "__main__":

    main()