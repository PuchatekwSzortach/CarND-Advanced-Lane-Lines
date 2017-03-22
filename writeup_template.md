##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[chessboard_original_image]: ./writeup_images/chessboard_original_image.jpeg
[chessboard_undistorted_image]: ./writeup_images/chessboard_undistorted_image.jpeg
[test_image]: ./writeup_images/test_image.jpg
[test_image_undistorted]: ./writeup_images/test_image_undistorted.jpg
[test_image_with_warp_mask]: ./writeup_images/test_image_with_warp_mask.jpg
[test_image_warped]: ./writeup_images/test_image_warped.jpg
[image_with_shadow]: ./writeup_images/image_with_shadow.jpeg
[image_with_shadow_removed]: ./writeup_images/image_with_shadow_removed.jpeg
[image_with_shadow_mask]: ./writeup_images/image_with_shadow_mask.jpeg
[image_with_shadow_removed_mask]: ./writeup_images/image_with_shadow_removed_mask.jpeg

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code for camera calibration is contained in `script/calibrate.camera.py` in function `compute_camera_calibration()` starting on line `19`. First a 3D grid of (x, y, z) size of (9, 6, 1) is prepared to represent true coordinates of chessboard square corners, called object points below. Z-value is set to 0 for all entries, since all points lie on the same z-plane.

Then `cv2.findChessboardCorners()` is used to find corners of chessboard corners in calibration images.

Calibration is performed with `cv2.calibrateCamera()` based on object space and image space points pairs and resulting camera matrix and distortion coefficients are stored for later use.

`script/calibrate.camera.py` also contains `undistort_sample_image()` function on line 71 that undistorts an input image based on previously computed camera parameters. A sample input and undistorted output are presented below:

![chessboard_original_image] ![chessboard_undistorted_image]

###Pipeline (single images)

`scripts/show_preprocessing_pipeline.py` demonstrates different stages of preprocessing pipeline.
Images created below were obtained with function `show_preprocessing_pipeline_for_test_images()` starting on line 18. Actual preprocessing code is contained inside `car.processing.ImageProcessor` class.

####1. Provide an example of a distortion-corrected image.

As a first step of the pipeline I compute undistorted image based on previously established camera calibration. Below is an example of original image and its undistorted counterpart.

![test_image] ![test_image_undistorted]

As can be seen undistortion doesn't affect image too much. This is quite expected, distortion effects are only significant around image edges (more specifically, lense edges), but our are of interest is dead in the middle of the camera. Pay attention to car hood though, you can see subtle changes to its shape between original and undistorted image.

Undistorted image was computed with `car.processing.ImageProcessor::get_undistorted_image()`, which perfoms a simple call to `cv2.undistort()`.


####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As a second step of my preprocessing pipeline I warped image so as to give a bird-eye view on area likely occupied by the lane. This has several advantages:  
- making lane lines appear more straight even when they are turning
- making lane lines width more uniform across whole image  
- making it easier to mask out areas outside of the lane

Below is a sample image with area to be warped marked in original coordinates, as well as the same image after warping

![test_image_with_warp_mask] ![test_image_warped]

Source and destination points for warping are specified in `car.processing.get_perspective_transformation_source_coordinates()` and `car.processing.get_perspective_transformation_destination_coordinates()` on lines 267 and 279, respecively. Source and destination points are chosen so as to map likely lane area to a rectangle spanning middle band of destination image.

Warped image is computed with `car.processing.ImageProcessor::get_warped_image()` on line 111, which makes a simple call to `cv2.warpPerspective()`.

####3. Shadow removal

Recongnizing that shadow areas affected my binary mask images (shown later), I added a simple shadow removal method based on a paper *"A robust approach for road detection with shadow removal technique"* by Salim, Cheng and Degui. For implementation details please refer to the paper, but a quick intuition is that shadows are areas that have high saturation (thus lively colors), but low value - so areas of lively colors that don't appear lively in the image. Once shadow areas are identified, shadows can be removed (or at least their impact attenuated) by making their moments closer to moments of surrounding non-shadow areas.

In practice above method works well on small to medium size shadow patches and proves useful in removing some of the shadows that happen to fall within car lanes.

While I perform shadow removal on warped images (`car.processing.ImageProcessor.get_preprocessed_image()`, line 125, below I present a sample unwarped image, first in original form then with shadow removed, followed by a saturation-based mask for both cases.

![image_with_shadow] 
![image_with_shadow_removed]
![image_with_shadow_mask]
![image_with_shadow_removed_mask]

As can be seen most of shadow right in front of car hood was removed, which simplifies further detection stages. A careful reader would note that black car in right lane also got removed from saturation mask - this isn't a problem in our scenario, but it highlights that above shadow removal method shoud be used with care.

####4. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

