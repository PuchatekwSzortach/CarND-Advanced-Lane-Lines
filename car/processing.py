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

        saturation = hls[:, :, 2]

        lower_threshold = self.parameters['saturation_thresholds'][0]
        upper_threshold = self.parameters['saturation_thresholds'][1]

        return np.uint8((lower_threshold <= saturation) & (saturation <= upper_threshold))

    def get_x_direction_gradient_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        x_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size))
        x_gradient = 255 * x_gradient / np.max(x_gradient)

        lower_threshold = self.parameters['x_gradient_thresholds'][0]
        upper_threshold = self.parameters['x_gradient_thresholds'][1]

        return np.uint8((lower_threshold <= x_gradient) & (x_gradient <= upper_threshold))

    def get_y_direction_gradient_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size))
        gradient = 255 * gradient / np.max(gradient)

        lower_threshold = self.parameters['y_gradient_thresholds'][0]
        upper_threshold = self.parameters['y_gradient_thresholds'][1]

        return np.uint8((lower_threshold <= gradient) & (gradient <= upper_threshold))

    def get_gradient_magnitude_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        x_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size))
        x_gradient = 255 * x_gradient / np.max(x_gradient)

        y_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size))
        y_gradient = 255 * y_gradient / np.max(y_gradient)

        magnitude = np.sqrt((x_gradient ** 2) + (y_gradient ** 2))

        lower_threshold = self.parameters['gradient_magnitude_thresholds'][0]
        upper_threshold = self.parameters['gradient_magnitude_thresholds'][1]

        return np.uint8((lower_threshold <= magnitude) & (magnitude <= upper_threshold))

    def get_preprocessed_image(self, image):
        """
        Get preprocessed image
        :param image:
        :return: binary image
        """

        undistorted_image = self.get_undistorted_image(image)

        saturation = self.get_saturation_mask(undistorted_image)
        x_gradient = self.get_x_direction_gradient_mask(undistorted_image)

        y_gradient = self.get_y_direction_gradient_mask(undistorted_image)
        magnitude = self.get_gradient_magnitude_mask(undistorted_image)

        binary = saturation | x_gradient | (y_gradient & magnitude)

        kernel = np.ones((5, 1))
        binary = cv2.erode(binary, kernel=kernel)
        binary = cv2.dilate(binary, kernel=kernel)

        return binary

    def get_preprocessed_image_for_video(self, image):

        mask = self.get_preprocessed_image(image)

        return 255 * np.dstack([mask, mask, mask])


class ShadowPreprocessor:
    """
    Class for removing shadows
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

    def get_shadow(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        shadow = (saturation + 1) / (value + 1)
        return shadow

    def get_shadow_mask(self, image):

        shadow = self.get_shadow(image)
        shadow = shadow.astype(np.uint8)
        _, mask = cv2.threshold(shadow, thresh=0, maxval=1, type=cv2.THRESH_BINARY)

        kernel = np.ones((5, 5))
        mask = cv2.erode(mask, kernel=kernel)
        mask = cv2.dilate(mask, kernel=kernel)

        return mask

    def get_shadow_blobs(self, image):

        mask = self.get_shadow_mask(image)

        value, data = cv2.connectedComponents(mask)

        blobs = []

        for index in range(1, value + 1):

            if np.sum(data == index) > 5000:

                blob = np.zeros_like(mask)
                blob[data == index] = 1

                blobs.append(blob)

        return blobs

    def get_image_without_shadows(self, image):

        blobs = self.get_shadow_blobs(image)

        all_shadows_mask = np.zeros(shape=image.shape[:2])
        all_shadows_mask += np.sum(np.array(blobs), axis=0)

        all_shadows_mask = all_shadows_mask > 0

        no_shadows_mask = 1 - all_shadows_mask
        reconstructed_image = np.zeros_like(image)
        reconstructed_image[no_shadows_mask == 1] = image[no_shadows_mask == 1]

        for blob in blobs:

            kernel = np.ones((9, 9))
            outer_area_mask = cv2.dilate(blob, kernel) - blob

            outer_area_mean = np.mean(image[outer_area_mask == 1])
            outer_area_std = np.std(image[outer_area_mask == 1])

            shadow_area_mean = np.mean(image[blob == 1])
            shadow_area_std = np.std(image[blob == 1])

            local_reconstruction = np.zeros_like(image)

            numerator = (image[blob == 1] - shadow_area_mean) * shadow_area_std
            local_reconstruction[blob == 1] = outer_area_mean + (numerator / outer_area_std)

            reconstructed_image += local_reconstruction

        return reconstructed_image