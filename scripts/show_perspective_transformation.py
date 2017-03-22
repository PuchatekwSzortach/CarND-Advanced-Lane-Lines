"""
A script that shows perspective transformation of road surface
"""

import glob
import os

import cv2
import vlogging

import car.utilities
import car.config
import car.processing


def show_tranformed_test_images(logger):

    paths = glob.glob(os.path.join(car.config.test_images_directory, "*.jpg"))

    for path in paths:

        image = car.utilities.get_image(path)

        source = car.processing.get_perspective_transformation_source_coordinates(image.shape)
        destination = car.processing.get_perspective_transformation_destination_coordinates(image.shape)

        matrix = cv2.getPerspectiveTransform(source, destination)

        warped = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        images = [image, warped]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        target_size = (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5))
        logger.info(vlogging.VisualRecord("Image, processed",
                                          [cv2.resize(image, target_size) for image in images], fmt='jpg'))


def main():

    logger = car.utilities.get_logger(car.config.log_path)

    show_tranformed_test_images(logger)


if __name__ == "__main__":

    main()