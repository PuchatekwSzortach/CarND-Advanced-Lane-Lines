"""
Configuration file
"""

log_path = "/tmp/advanced_lanes_detection.html"

calibration_pickle_path = "./calibration_pickle.p"
calibration_images_directory = "./camera_cal/"

test_images_directory = "./test_images"
additional_test_images_directory = "./test_images_video_all"

my_car_test_images_directory = "./test_images_my_car"

video_output_directory = "../../data/advanced_lanes_detection/"

# Based on typical lane dimensions and cropping used in image transformation
metres_per_pixel_width = 3.7 / 400
metres_per_pixel_height = 30 / 820
