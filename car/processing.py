"""
Module with image processing code
"""

import pickle

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Class for preprocessing images to make task of lane finding easier
    """

    def __init__(self, calibration_pickle_path, parameters):
        """
        Constructor
        :param calibration_pickle_path: path to pickle with camera calibration data
        :param parameters: dictionary with parameters for various preprocessing stages
        """

        with open(calibration_pickle_path, "rb") as file:

            data = pickle.load(file)

            self.camera_matrix = data['camera_matrix']
            self.distortion_coefficients = data['distortion_coefficients']

        self.parameters = parameters

    def get_undistorted_image(self, image):

        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)

    def get_cropped_image(self, image):

        top_margin = self.parameters['cropping_margins'][0][0]
        bottom_margin = self.parameters['cropping_margins'][0][1]

        left_margin = self.parameters['cropping_margins'][1][0]
        right_margin = self.parameters['cropping_margins'][1][1]

        return image[
               top_margin:image.shape[0] - bottom_margin,
               left_margin:image.shape[1] - right_margin, :]

    def get_saturation_mask(self, image):
        """
        Return masked based on saturation channel of an HSL colorspace image
        :param image: RGB image
        :return: saturation mask
        """

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)

        lumination = hls[:, :, 1]
        saturation = hls[:, :, 2]

        lower_threshold = self.parameters['saturation_thresholds'][0] * (lumination + 10) / 120
        upper_threshold = self.parameters['saturation_thresholds'][1]

        mask = np.zeros_like(saturation, dtype=np.uint8)

        lumination_threshold = 100

        mask[(lumination > lumination_threshold) & (lower_threshold <= saturation) & (saturation <= upper_threshold)] = 1
        mask[(lumination < lumination_threshold) & (saturation < 25)] = 1

        return mask

    def get_x_direction_gradient_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        x_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size))
        x_gradient = 255 * x_gradient / np.max(x_gradient)

        lower_threshold = self.parameters['x_gradient_thresholds'][0]
        upper_threshold = self.parameters['x_gradient_thresholds'][1]

        return np.uint8((lower_threshold <= x_gradient) & (x_gradient <= upper_threshold))

    def get_preprocessed_image(self, image):
        """
        Get preprocessed image
        :param image:
        :return: binary image
        """

        undistorted_image = self.get_undistorted_image(image)

        saturation = self.get_saturation_mask(undistorted_image)
        x_gradient = self.get_x_direction_gradient_mask(undistorted_image)

        binary = saturation | x_gradient
        return binary

    def get_preprocessed_image_for_video(self, image):

        mask = self.get_preprocessed_image(image)

        return 255 * np.dstack([mask, mask, mask])

