import cv2
import numpy as np

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS, Open3dVisualizer

model_path = "models/CenterSnap_sim.onnx"
poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"

# Read rgb image and depth map
# Download data from the original repository: https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz
img_path = "test/scene_2/0073_color.png"
depth_path = "test/scene_2/0073_depth.png"
rgb_img, depth_norm, actual_depth = load_img_NOCS(img_path, depth_path)

# Initialize pose estimator with autoencoder
poseEstimator = CenterSnap(model_path, poincloud_estimator_path)

# Initialize the Open3d visualizer
open3dVisualizer = Open3dVisualizer()

# Update pose estimator
ret = poseEstimator(rgb_img, depth_norm)

if ret:
	# Draw RGB image with 2d data
	combined_img = poseEstimator.draw_points_2d(rgb_img)
	print(combined_img.shape)
	cv2.namedWindow("Projected Pose", cv2.WINDOW_NORMAL)
	cv2.imshow("Projected Pose", combined_img)

	# Draw 3D data
	open3dVisualizer(poseEstimator.points_3d_list, poseEstimator.boxes_3d_list, is_image=True)

	o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
	o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
	o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (640, 480))

	combined_img = cv2.hconcat([combined_img, o3d_screenshot_mat])
	cv2.imwrite("pose3d.png", combined_img)

	cv2.waitKey(0) 


