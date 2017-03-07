"""
Module with image processing code
"""

import pickle

import cv2


class ImagePreprocessor:
    """
    Class for preprocessing images to make task of lane finding easier
    """

    def __init__(self, calibration_pickle_path):
        """
        Constructor
        :param calibration_pickle_path: path to pickle with camera calibration data
        """

        with open(calibration_pickle_path, "rb") as file:

            data = pickle.load(file)

            self.camera_matrix = data['camera_matrix']
            self.distortion_coefficients = data['distortion_coefficients']

    def get_undistorted_image(self, image):

        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)