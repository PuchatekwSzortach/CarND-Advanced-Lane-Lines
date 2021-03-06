"""
Module with image processing code
"""

import pickle
import pprint
import collections

import cv2
import numpy as np
import scipy.signal
import vlogging


class LaneSearchError(Exception):

    pass


class ImagePreprocessor:
    """
    Class for preprocessing images to make task of lane finding easier
    """

    def __init__(self, calibration_pickle_path, parameters, warp_matrix):
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
        self.warp_matrix = warp_matrix

        self.shadow_preprocessor = ShadowPreprocessor(calibration_pickle_path, parameters)

    def get_undistorted_image(self, image):

        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)

    def get_saturation_mask(self, image):
        """
        Return masked based on saturation channel of an HSL colorspace image
        :param image: RGB image
        :return: saturation mask
        """

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)

        saturation = hls[:, :, 2]

        lower_threshold = self.parameters['saturation_thresholds'][0]
        upper_threshold = self.parameters['saturation_thresholds'][1]

        return np.uint8((lower_threshold <= saturation) & (saturation <= upper_threshold))

    def get_x_direction_gradient_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        x_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size))
        x_gradient = 255 * x_gradient / np.max(x_gradient)

        lower_threshold = self.parameters['x_gradient_thresholds'][0]
        upper_threshold = self.parameters['x_gradient_thresholds'][1]

        return np.uint8((lower_threshold <= x_gradient) & (x_gradient <= upper_threshold))

    def get_y_direction_gradient_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size))
        gradient = 255 * gradient / np.max(gradient)

        lower_threshold = self.parameters['y_gradient_thresholds'][0]
        upper_threshold = self.parameters['y_gradient_thresholds'][1]

        return np.uint8((lower_threshold <= gradient) & (gradient <= upper_threshold))

    def get_gradient_magnitude_mask(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kernel_size = self.parameters['x_gradient_kernel_size']

        x_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size))
        x_gradient = 255 * x_gradient / np.max(x_gradient)

        y_gradient = np.abs(cv2.Sobel(grayscale, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size))
        y_gradient = 255 * y_gradient / np.max(y_gradient)

        magnitude = np.sqrt((x_gradient ** 2) + (y_gradient ** 2))

        lower_threshold = self.parameters['gradient_magnitude_thresholds'][0]
        upper_threshold = self.parameters['gradient_magnitude_thresholds'][1]

        return np.uint8((lower_threshold <= magnitude) & (magnitude <= upper_threshold))

    def get_warped_image(self, image):

        return cv2.warpPerspective(image, self.warp_matrix, (image.shape[1], image.shape[0]))

    def get_preprocessed_image(self, image):
        """
        Get preprocessed image
        :param image:
        :return: binary image
        """

        undistorted_image = self.get_undistorted_image(image)

        warped = self.get_warped_image(undistorted_image)
        deshadowed = self.shadow_preprocessor.get_image_without_shadows(warped)

        saturation = self.get_saturation_mask(deshadowed)
        x_gradient = self.get_x_direction_gradient_mask(deshadowed)

        binary = saturation | x_gradient

        mask = self.get_cropping_mask(binary.shape)
        binary *= mask

        kernel = np.ones((5, 3))
        binary = cv2.erode(binary, kernel=kernel)
        binary = cv2.dilate(binary, kernel=kernel)

        return binary

    def get_preprocessed_image_for_video(self, image):

        mask = self.get_preprocessed_image(image)
        return 255 * np.dstack([mask, mask, mask])

    def get_cropping_mask(self, image_shape):
        """
        Returns a mask such that pixels outside of it can be ignored
        :param image_shape:
        :return: binary image
        """

        mask = np.zeros(shape=image_shape, dtype=np.uint8)

        mask_coordinates = [
            [image_shape[1] // 4, image_shape[0]],  # left bottom corner
            [3 * image_shape[1] // 4, image_shape[0]],  # right bottom corner
            [int(0.7 * image_shape[1]), 0],  # right upper corner
            [int(0.3 * image_shape[1]), 0],  # left upper corner
        ]

        cv2.fillPoly(mask, np.array([mask_coordinates]).astype(np.int32), color=1)

        return mask


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

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

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


def get_perspective_transformation_source_coordinates(image_shape):

    coordinates = np.array([
        [150, image_shape[0] - 20],  # left bottom
        [image_shape[1] - 150, image_shape[0] - 20],  # right bottom
        [image_shape[1] - 550, 440],  # right top
        [550, 440],  # left top
    ])

    return coordinates.astype(np.float32)


def get_perspective_transformation_destination_coordinates(image_shape):

    coordinates = np.array([
        [400, image_shape[0]],  # lower left corner
        [image_shape[1] - 400, image_shape[0]],  # lower right corner
        [image_shape[1] - 400, 0],  # upper right corner
        [400, 0]  # upper left corner
    ])

    return coordinates.astype(np.float32)


class LaneLineFinder:
    """
    Class for search for a lane line and computing its equation
    """

    def __init__(self, image, offset):

        self.image = image
        self.offset = offset

        self.kernel_width = 30
        self.kernel_height = 20

    def get_lane_starting_coordinates(self):

        # Compute vertical histogram to find x with largest response
        kernel = np.ones((self.image.shape[0], self.kernel_width))

        histogram = scipy.signal.convolve2d(self.image, kernel, mode='valid').squeeze().astype(np.int32)
        peak = np.argmax(histogram)

        x = peak + (kernel.shape[1] // 2)

        # For selected x compute y that has most white pixels
        column_image = self.image[:, x - (self.kernel_width//2): x + (self.kernel_width//2)]
        kernel = np.ones((self.kernel_height, column_image.shape[1]))

        histogram = scipy.signal.convolve2d(column_image, kernel, mode='valid').squeeze().astype(np.int32)
        peak = np.argmax(histogram)

        y = peak + (kernel.shape[0] // 2)

        return x, y

    def scan_image_for_line_candidates_without_prior_knowledge(self, x, y, direction):
        """
        Scan image above starting points for lane candidates
        :return: tuple (left area border, best hit, right area border), each element of the tuple is a list of points
        """

        kernel = np.ones((self.kernel_height, self.kernel_width))

        original_half_search_width = self.kernel_width
        current_half_search_width = original_half_search_width

        left_border_points = []
        candidate_points = []
        right_border_points = []

        continue_scan_conditions_map = {
            "up": lambda y, kernel_height: bool(y - kernel_height > 0),
            "down": lambda y, kernel_height: bool(y < self.image.shape[0])
        }

        update_ys_map = {

            "up": lambda y, kernel_height: y - kernel_height,
            "down": lambda y, kernel_height: y + kernel_height
        }

        continue_scan_condition = continue_scan_conditions_map[direction]
        update_y_condition = update_ys_map[direction]

        # If needed change y so that matching wouldn't lead outside of image coordinates
        y = np.clip(y, self.kernel_height, self.image.shape[0] - self.kernel_height)

        while continue_scan_condition(y, self.kernel_height) is True:

            left_band_limit = x - current_half_search_width
            right_band_limit = x + current_half_search_width

            left_border_points.append([left_band_limit, y])
            right_border_points.append([right_band_limit, y])

            candidate_band = self.image[y - self.kernel_height:y, left_band_limit: right_band_limit]

            response = scipy.signal.correlate2d(candidate_band, kernel, mode='valid').squeeze()
            max_response = np.max(response)

            if max_response > 100:

                x = left_band_limit + np.argmax(response) + (self.kernel_width // 2)
                candidate_points.append([x, y])

                current_half_search_width = original_half_search_width

            elif max_response == 0:

                current_half_search_width = 2 * original_half_search_width

            y = update_y_condition(y, self.kernel_height)

        return left_border_points, candidate_points, right_border_points

    def scan_image_for_line_candidates_using_prior_knowledge(self, recent_lane_equations):
        """
        Scan image above starting points for lane candidates
        :return: tuple (left area border, best hit, right area border), each element of the tuple is a list of points
        """

        kernel = np.ones((self.kernel_height, self.kernel_width))

        half_search_width = self.kernel_width

        left_border_points = []
        candidate_points = []
        right_border_points = []

        # If needed change y so that matching wouldn't lead outside of image coordinates
        y = self.kernel_height

        while y < self.image.shape[0]:

            old_x_center = self.get_x(recent_lane_equations[-1], y) - self.offset
            left_band_limit = old_x_center - half_search_width
            right_band_limit = old_x_center + half_search_width

            left_border_points.append([left_band_limit, y])
            right_border_points.append([right_band_limit, y])

            candidate_band = self.image[y - self.kernel_height:y, left_band_limit: right_band_limit]

            response = scipy.signal.correlate2d(candidate_band, kernel, mode='valid').squeeze()
            max_response = np.max(response)

            if max_response > 100:

                x = left_band_limit + np.argmax(response) + (self.kernel_width // 2)
                candidate_points.append([x, y])

            y += self.kernel_height

        return left_border_points, candidate_points, right_border_points

    def get_lane_search_image_without_prior_knowledge(self):

        start_x, start_y = self.get_lane_starting_coordinates()

        search_image = np.zeros(shape=(self.image.shape + (3,)))
        search_image[:, :, 0] = 255 * self.image

        # Draw starting point
        cv2.circle(search_image, (start_x, start_y), radius=15, color=(0, 255, 0), thickness=-1)

        # Scan through image for lane lines
        upper_left_search_border_points, upper_center_points, upper_right_search_border_points = \
            self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="up")

        lower_left_search_border_points, lower_center_points, lower_right_search_border_points = \
            self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="down")

        # Draw search area and best fit points
        cv2.polylines(search_image,
                      np.int32([list(reversed(lower_left_search_border_points)) + upper_left_search_border_points]),
                      isClosed=False, color=(0, 0, 200), thickness=4)

        cv2.polylines(search_image,
                      np.int32([list(reversed(lower_right_search_border_points)) + upper_right_search_border_points]),
                      isClosed=False, color=(0, 0, 200), thickness=4)

        cv2.polylines(search_image,
                      np.int32([list(reversed(lower_center_points)) + upper_center_points]),
                      isClosed=False, color=(0, 200, 0), thickness=4)

        return search_image

    def get_lane_equation_without_prior_knowledge(self):

        start_x, start_y = self.get_lane_starting_coordinates()

        # Scan through image for lane lines
        _, upper_center_points, _ = \
            self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="up")

        _, lower_center_points, _ = \
            self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="down")

        center_points = np.array(list(reversed(lower_center_points)) + upper_center_points)

        if len(center_points) == 0:
            raise LaneSearchError()

        return np.polyfit(center_points[:, 1], center_points[:, 0] + self.offset, deg=2)

    def get_lane_search_image_using_prior_knowledge(self, recent_lane_equations):

        search_image = np.zeros(shape=(self.image.shape + (3,)))
        search_image[:, :, 0] = 255 * self.image

        # Scan through image for lane lines
        left_search_border_points, center_points, right_search_border_points = \
            self.scan_image_for_line_candidates_using_prior_knowledge(recent_lane_equations)

        # Draw search area and best fit points
        cv2.polylines(search_image,
                      np.int32([left_search_border_points]),
                      isClosed=False, color=(0, 0, 200), thickness=4)

        cv2.polylines(search_image,
                      np.int32([right_search_border_points]),
                      isClosed=False, color=(0, 0, 200), thickness=4)

        cv2.polylines(search_image,
                      np.int32([center_points]),
                      isClosed=False, color=(0, 200, 0), thickness=4)

        return search_image

    def get_lane_equation_using_prior_knowledge(self, recent_lane_equations, logger):

        # Scan through image for lane lines
        left_search_border_points, center_points, right_search_border_points = \
            self.scan_image_for_line_candidates_using_prior_knowledge(recent_lane_equations)

        # If we got a very bad fit, resort to search from scratch
        if len(center_points) < 5:

            start_x, start_y = self.get_lane_starting_coordinates()

            # Scan through image for lane lines
            _, upper_center_points, _ = \
                self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="up")

            _, lower_center_points, _ = \
                self.scan_image_for_line_candidates_without_prior_knowledge(start_x, start_y, direction="down")

            center_points = list(reversed(lower_center_points)) + upper_center_points

        if len(center_points) == 0:

            search_image = self.get_lane_search_image_without_prior_knowledge()
            images = [255 * self.image, search_image]

            logger.info(vlogging.VisualRecord(
                "Failed get_lane_equation_using_prior_knowledge()", images, str(self.image.shape)))

            return recent_lane_equations[-1]

        center_points = np.array(center_points)
        latest_equation = np.polyfit(center_points[:, 1], center_points[:, 0] + self.offset, deg=2)

        all_equations = collections.deque(recent_lane_equations.copy(), maxlen=len(recent_lane_equations) + 1)
        all_equations.append(latest_equation)

        return np.mean(all_equations, axis=0)

    def get_lane_equation(self, recent_lane_equations=(), logger=None):

        if len(recent_lane_equations) == 0:

            return self.get_lane_equation_without_prior_knowledge()

        else:

            return self.get_lane_equation_using_prior_knowledge(recent_lane_equations, logger)

    def get_x(self, lane_equation, y):

        return int((lane_equation[0] * (y**2)) + (lane_equation[1] * y) + lane_equation[2])


def get_lane_mask(image, lane_equation, warp_matrix):

    mask = np.zeros(shape=image.shape[:2])

    arguments = np.linspace(mask.shape[0], 0)
    values = (lane_equation[0] * (arguments ** 2)) + (lane_equation[1] * arguments) + lane_equation[2]

    points = list(zip(values, arguments))
    cv2.polylines(mask, np.int32([points]), isClosed=False, color=1, thickness=20)

    return cv2.warpPerspective(mask, warp_matrix, (mask.shape[1], mask.shape[0]))


def draw_lane(image, left_line_equation, right_line_equation, warp_matrix):

    ys = np.linspace(0, image.shape[0])
    left_xs = (left_line_equation[0] * (ys**2)) + (left_line_equation[1] * ys) + left_line_equation[2]
    right_xs = (right_line_equation[0] * (ys ** 2)) + (right_line_equation[1] * ys) + right_line_equation[2]

    left_polyline = list(zip(left_xs, ys))
    right_polyline = list(zip(right_xs, ys))

    polyline = left_polyline + list(reversed(right_polyline))

    lane_image = np.zeros_like(image)
    cv2.fillPoly(lane_image, np.int32([polyline]), color=(0, 255, 0))

    warped_lane_image = cv2.warpPerspective(lane_image, warp_matrix, (image.shape[1], image.shape[0]))

    return np.clip(image + warped_lane_image, 0, 255)


class LaneStatisticsComputer:

    def __init__(self, image_shape, metres_per_pixel_width, metres_per_pixel_height):

        self.image_shape = image_shape
        self.metres_per_pixel_width = metres_per_pixel_width
        self.metres_per_pixel_height = metres_per_pixel_height

    def get_line_curvature(self, line_equation):

        y = self.image_shape[0]

        # Line equation coefficients
        a = line_equation[0]
        b = line_equation[1]

        first_derivative = (2 * a * y * self.metres_per_pixel_height) + b
        second_derivative = 2 * a

        radius = np.power(1 + np.square(first_derivative), 3/2) / np.abs(second_derivative)
        return "{}m radius".format(int(radius))

    def get_lane_displacement(self, left_line_equation, right_line_equation):

        y = self.image_shape[0]

        left_x = (left_line_equation[0] * (y**2)) + (left_line_equation[1] * y) + left_line_equation[2]
        right_x = (right_line_equation[0] * (y ** 2)) + (right_line_equation[1] * y) + right_line_equation[2]

        lane_center_x = (right_x + left_x) / 2
        middle_x = self.image_shape[1] / 2

        pixels_displacement = abs(middle_x - lane_center_x)

        return "{:.2f} m".format(self.metres_per_pixel_width * pixels_displacement)


class SimpleVideoProcessor:
    """
    Video processor that searches for lane lines. Each frame is processed separately.
    """

    def __init__(self, preprocessor, statistics_computer, source_points, destination_points):

        self.preprocessor = preprocessor
        self.statistics_computer = statistics_computer

        self.warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.unwarp_matrix = cv2.getPerspectiveTransform(destination_points, source_points)

    def get_image_with_lanes(self, image):

        undistorted_image = self.preprocessor.get_undistorted_image(image)
        mask = self.preprocessor.get_preprocessed_image(image)

        left_finder = LaneLineFinder(mask[:, :mask.shape[1] // 2], offset=0)
        right_finder = LaneLineFinder(mask[:, (mask.shape[1] // 2):], offset=mask.shape[1] // 2)

        try:

            left_lane_equation = left_finder.get_lane_equation()
            right_lane_equation = right_finder.get_lane_equation()

            left_lane_mask = get_lane_mask(undistorted_image, left_lane_equation, self.unwarp_matrix)
            right_lane_mask = get_lane_mask(undistorted_image, right_lane_equation, self.unwarp_matrix)

            image_with_lanes = undistorted_image.copy().astype(np.float32)

            image_with_lanes[left_lane_mask == 1] = (0, 255, 0)
            image_with_lanes[right_lane_mask == 1] = (0, 255, 0)

            left_curvature = self.statistics_computer.get_line_curvature(left_lane_equation)
            right_curvature = self.statistics_computer.get_line_curvature(right_lane_equation)
            displacement = self.statistics_computer.get_lane_displacement(left_lane_equation, right_lane_equation)

            cv2.putText(image_with_lanes, "Left lane curvature: {}".format(left_curvature), (100, 80),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

            cv2.putText(image_with_lanes, "Right lane curvature: {}".format(right_curvature), (100, 120),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

            cv2.putText(image_with_lanes, "Displacement from lane center: {}".format(displacement), (100, 160),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

            return image_with_lanes

        except LaneSearchError:

            print("LaneSearchError")

            return image


class SmoothVideoProcessor:
    """
    Video processor that searches for lane lines. A history of recent results is used when searching for lane lines
    """

    def __init__(self, preprocessor, statistics_computer, source_points, destination_points, logger):

        self.preprocessor = preprocessor
        self.statistics_computer = statistics_computer

        self.warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.unwarp_matrix = cv2.getPerspectiveTransform(destination_points, source_points)

        self.left_lane_fits = collections.deque(maxlen=2)
        self.right_lane_fits = collections.deque(maxlen=2)

        self.logger = logger

    def get_image_with_lanes(self, image):

        undistorted_image = self.preprocessor.get_undistorted_image(image)
        mask = self.preprocessor.get_preprocessed_image(image)

        left_finder = LaneLineFinder(mask[:, :mask.shape[1] // 2], offset=0)
        right_finder = LaneLineFinder(mask[:, (mask.shape[1] // 2):], offset=mask.shape[1] // 2)

        left_lane_equation = left_finder.get_lane_equation(self.left_lane_fits, logger=self.logger)
        right_lane_equation = right_finder.get_lane_equation(self.right_lane_fits, logger=self.logger)

        self.left_lane_fits.append(left_lane_equation)
        self.right_lane_fits.append(right_lane_equation)

        image_with_lanes = undistorted_image.copy().astype(np.float32)
        image_with_lanes = draw_lane(image_with_lanes, left_lane_equation, right_lane_equation, self.unwarp_matrix)

        left_curvature = self.statistics_computer.get_line_curvature(left_lane_equation)
        right_curvature = self.statistics_computer.get_line_curvature(right_lane_equation)
        displacement = self.statistics_computer.get_lane_displacement(left_lane_equation, right_lane_equation)

        cv2.putText(image_with_lanes, "Left lane curvature: {}".format(left_curvature), (100, 80),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        cv2.putText(image_with_lanes, "Right lane curvature: {}".format(right_curvature), (100, 120),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        cv2.putText(image_with_lanes, "Displacement from lane center: {}".format(displacement), (100, 160),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))

        return image_with_lanes
