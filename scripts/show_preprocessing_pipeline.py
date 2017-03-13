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


def show_preprocessing_pipeline_for_additional_test_images(logger):

    paths = glob.glob(os.path.join(car.config.additional_test_images_directory, "*.jpg"))

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


def show_preprocessing_pipeline_for_shadow_images(logger):

    paths = ["test_06.jpg", "test_07.jpg", "test_09.jpg", "test_10.jpg", "test_12.jpg"]

    paths = [os.path.join(car.config.additional_test_images_directory, path) for path in paths]

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
        logger.info(vlogging.VisualRecord("Image, saturation, x_gradient, processed",
                                          [cv2.resize(image, target_size) for image in images]))


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


def get_additional_test_frames(logger):

    parameters = {
        "cropping_margins": [[350, 50], [100, 100]],
        "saturation_thresholds": [100, 255],
        "x_gradient_thresholds": [30, 255],
        "x_gradient_kernel_size": 9
    }

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters)

    path = "./project_video.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    frame = clip.get_frame(t=22.2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=39.3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=42)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    path = "./challenge_video.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    frame = clip.get_frame(t=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=4.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=7)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    path = "./harder_challenge_video.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    frame = clip.get_frame(t=7.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=14.3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=22.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=23)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=27)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))

    frame = clip.get_frame(t=40)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = preprocessor.get_preprocessed_image_for_video(frame)
    logger.info(vlogging.VisualRecord("Movie frame", [frame, processed_frame], fmt="jpg"))


def main():

    logger = car.utilities.get_logger(car.config.log_path)
    show_preprocessing_pipeline_for_test_images(logger)
    # show_preprocessing_pipeline_for_additional_test_images(logger)

    # show_preprocessing_pipeline_for_shadow_images(logger)

    # show_preprocessing_pipeline_for_test_videos()
    # get_additional_test_frames(logger)


if __name__ == "__main__":

    main()