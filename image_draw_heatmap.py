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

# Draw object position heat map
combined_img = poseEstimator.draw_heatmap(rgb_img, alpha=0.5)

cv2.imwrite("heatmap.png", combined_img)

cv2.namedWindow("Heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("Heatmap", combined_img)
cv2.waitKey(0)


