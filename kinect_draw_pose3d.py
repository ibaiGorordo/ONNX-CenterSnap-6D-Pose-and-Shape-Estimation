import cv2
import numpy as np
import sys

sys.path.insert(1, 'pyKinectAzure')
import pykinect_azure as pykinect

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS, Open3dVisualizer

KINECT_MAT_720 = np.array([[613.99*1.5   ,   0.     , 637.48   ,   0.     ],
       					   [  0.     , 613.94*1.5   , 368.45   ,   0.     ],
       					   [  0.     ,   0.     ,   1.     ,   0.     ],
       					   [  0.     ,   0.     ,   0.     ,   1.     ]])

model_path = "models/CenterSnap_sim.onnx"
poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"

# Initialize pose estimator with autoencoder
poseEstimator = CenterSnap(model_path, poincloud_estimator_path, camera_mat=KINECT_MAT_720)

# Initialize the Open3d visualizer
open3dVisualizer = Open3dVisualizer()

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)
# print(device.calibration)

cv2.namedWindow('Projected Pose',cv2.WINDOW_NORMAL)
while True:

	# Get capture
	capture = device.update()

	# Get the color image from the capture
	ret, rgb_img = capture.get_color_image()

	if not ret:
		continue

	# Get the colored depth
	ret, transformed_depth = capture.get_transformed_depth_image()

	# Update pose estimator
	ret = poseEstimator(rgb_img, transformed_depth/255.0)

	if ret:

		# Draw RGB image with 2d data
		combined_img = poseEstimator.draw_points_2d(rgb_img)[:,:,:3]

		# Draw 3D data
		open3dVisualizer(poseEstimator.points_3d_list, poseEstimator.boxes_3d_list)
	else:
		combined_img = rgb_img[:,:,:3]

	# Convert Open3D map to image
	o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
	o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
	o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (1280, 720))  

	combined_img = cv2.hconcat([combined_img, o3d_screenshot_mat])

	# combined_img = poseEstimator.draw_depthmap(rgb_img, alpha=0, max_dist=1)
	cv2.namedWindow("Projected Pose", cv2.WINDOW_NORMAL)
	cv2.imshow("Projected Pose", combined_img)

	# Press q key to stop
	if cv2.waitKey(1) == ord('q'): 
		break




