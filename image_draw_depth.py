import cv2

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS

model_path = "models/CenterSnap_sim.onnx"

# Read rgb image and depth map
# Download data from the original repository: https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz
img_path = "test/scene_2/0073_color.png"
depth_path = "test/scene_2/0073_depth.png"
rgb_img, depth_norm, actual_depth = load_img_NOCS(img_path, depth_path)

# Initialize pose estimator
poseEstimator = CenterSnap(model_path)

# Update pose estimator
poseEstimator(rgb_img, depth_norm)

# Draw Depth map
combined_img = poseEstimator.draw_depthmap(rgb_img, max_dist = 2, alpha=0.4)

cv2.imwrite("depthmap.png", combined_img)

cv2.namedWindow("Depth map", cv2.WINDOW_NORMAL)
cv2.imshow("Depth map", combined_img)
cv2.waitKey(0)


