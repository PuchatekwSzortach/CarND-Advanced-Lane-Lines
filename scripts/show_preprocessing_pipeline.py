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

    image_shape = (720, 1280)
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters, warp_matrix)

    for path in paths:

        image = car.utilities.get_image(path)

        image_with_warp_mask = preprocessor.get_undistorted_image(image)
        cv2.polylines(image_with_warp_mask, np.int32([source]), isClosed=True, color=(0, 0, 255), thickness=4)

        warped = preprocessor.get_warped_image(image)
        processed = preprocessor.get_preprocessed_image(image)
        processed_deshadowed = preprocessor.get_preprocessed_image(image)

        images = [
            cv2.cvtColor(image_with_warp_mask, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), 255 * processed, 255 * processed_deshadowed]

        target_size = (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5))
        logger.info(vlogging.VisualRecord("Image, warped, saturation, x_gradient, processed",
                                          [cv2.resize(image, target_size) for image in images], fmt='jpg'))


def show_preprocessing_pipeline_for_test_videos():

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

    image_shape = (720, 1280)
    source = car.processing.get_perspective_transformation_source_coordinates(image_shape)
    destination = car.processing.get_perspective_transformation_destination_coordinates(image_shape)
    warp_matrix = cv2.getPerspectiveTransform(source, destination)

    preprocessor = car.processing.ImagePreprocessor(car.config.calibration_pickle_path, parameters, warp_matrix)

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
    # show_preprocessing_pipeline_for_test_images(logger)

    show_preprocessing_pipeline_for_test_videos()
    # get_additional_test_frames(logger)


if __name__ == "__main__":

    main()