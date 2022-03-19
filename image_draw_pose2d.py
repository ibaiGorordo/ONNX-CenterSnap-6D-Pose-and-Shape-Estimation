import cv2
import sys

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS

model_path = "models/CenterSnap_sim.onnx"
poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"

# Read rgb image and depth map
# Download data from the original repository: https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz
img_path = "test/scene_2/0073_color.png"
depth_path = "test/scene_2/0073_depth.png"
rgb_img, depth_norm, actual_depth = load_img_NOCS(img_path, depth_path)

# Initialize pose estimator
poseEstimator = CenterSnap(model_path, poincloud_estimator_path, min_conf=0.5)

# Update pose estimator
ret = poseEstimator(rgb_img, depth_norm)

if ret:

	# Draw projected points and boxes into the rgb image
	combined_img = poseEstimator.draw_points_2d(rgb_img)

	cv2.imwrite("pose2d.png", combined_img)

	cv2.namedWindow("Projected points", cv2.WINDOW_NORMAL)
	cv2.imshow("Projected points", combined_img)
	cv2.waitKey(0)


