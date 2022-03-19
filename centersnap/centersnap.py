import cv2
import onnxruntime
import numpy as np
import open3d as o3d

from .utils import *
from .centersnap_ae import CenterSnapAE

class CenterSnap():

	def __init__(self, model_path, autoencoder_model_path=None, camera_mat=NOCS_CAMERA_MAT, min_conf=0.5):

		# Initialize model
		self.initialize_model(model_path, autoencoder_model_path, camera_mat, min_conf)

	def __call__(self, image, depth_map):
		return self.estimate_pose(image, depth_map)

	def initialize_model(self, model_path, autoencoder_model_path=None, camera_mat=NOCS_CAMERA_MAT, min_conf=0.5):

		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
		self.min_conf = min_conf
		self.camera_mat = camera_mat

		# Initialize autoencoder to estimate the object pointcloud if available
		self.poincloudEstimator = None
		if (autoencoder_model_path):
			self.poincloudEstimator = CenterSnapAE(autoencoder_model_path)

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_pose(self, image, depth_map):

		input_tensor = self.prepare_input(image, depth_map)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		return self.process_output(outputs)

	def prepare_input(self, rgb_img, depth_map):

		self.img_height, self.img_width = rgb_img.shape[:2]

		# Resize input images
		rgb_img_res = cv2.resize(rgb_img[:,:,:3], (self.input_width,self.input_height))  
		depth_map_res = cv2.resize(depth_map, (self.input_width,self.input_height),  interpolation = cv2.INTER_NEAREST) 

		# rgb_img_res = cv2.cvtColor(rgb_img_res, cv2.COLOR_RGB2BGR)

		img_input = np.zeros((4, self.input_height, self.input_width), dtype=np.float32)

		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		rgb_img_res = ((rgb_img_res.astype(np.float32)/ 255.0 - mean) / std)
		rgb_img_res = rgb_img_res.transpose(2, 0, 1)

		img_input[0:3, :] = rgb_img_res
		img_input[3, :] = depth_map_res.astype(np.float32)
		img_input = img_input[np.newaxis,:,:,:]   

		return img_input

	def inference(self, input_tensor):

		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

		# print(time.time() - start)
		return outputs

	def process_output(self, output): 

		seg_pred, depth_pred, small_depth_pred, \
		pose_heatmap, pose_latent_emb, abs_pose = output

		# Process outputs
		self.seg_map = np.argmax(seg_pred[0], axis=0)
		self.depth_map = depth_pred[0]*255.0
		self.small_depth_map = small_depth_pred[0]*255.0
		self.heatmap = pose_heatmap[0]

		# Convert and filter pose data 
		self.latent_embeddings, self.abs_pose_outputs, \
		self.peak_img, self.scores,	self.indices = compute_pointclouds_and_poses(pose_heatmap[0], 
																				pose_latent_emb[0].transpose(1, 2, 0)/ 100.0, 
																				abs_pose[0].transpose(1, 2, 0)/ 100.0, 
																				self.min_conf)
		# Get the object labels from the segmentation map
		self.label_ids = [self.seg_map[index[0],index[1]] for index in self.indices]	

		if(not self.poincloudEstimator):
			return

		# Get 3D pose and pointcloud data
		self.num_objects = len(self.latent_embeddings)
		points_3d_list = []
		boxes_3d_list = []
		axes_3d_list = []
		for object_num in range(self.num_objects):
			# Use the autoencoder to estimate the pointcloud
			pointcloud = self.poincloudEstimator(self.latent_embeddings[object_num])
			rotated_pc, rotated_box, transformed_axes, _ = get_pointclouds_3d(self.abs_pose_outputs[object_num], pointcloud)
			
			points_3d_list.append(rotated_pc)
			boxes_3d_list.append(rotated_box)
			axes_3d_list.append(transformed_axes)

		self.points_3d_list, self.boxes_3d_list, self.axes_3d_list = points_3d_list,\
																	 boxes_3d_list,\
																	 axes_3d_list

		# Project 3D points and boxes into the 2D camera plane
		points_2d_list = []
		boxes_2d_list = []
		axes_2d_list = []
		for object_num in range(self.num_objects):
			points_2d_mesh, box_2d, projected_axes = get_pointclouds_2d(points_3d_list[object_num],
																		boxes_3d_list[object_num],
																		axes_3d_list[object_num],
																		self.camera_mat)
			
			points_2d_list.append(points_2d_mesh)
			boxes_2d_list.append(box_2d)
			axes_2d_list.append(projected_axes)

		self.points_2d_list, self.boxes_2d_list, self.axes_2d_list = points_2d_list,\
																	 boxes_2d_list,\
																	 axes_2d_list

		return len(points_3d_list) > 0													

	def draw_segmentation(self, image, alpha = 0.5):

		return util_draw_seg(self.seg_map, image, alpha)

	def draw_depthmap(self, image, max_dist = 2, alpha = 0.5):

		return util_draw_depth(self.depth_map, image, max_dist, alpha)

	def draw_small_depthmap(self, image, max_dist = None, alpha = 0.5):

		return util_draw_depth(self.small_depth_map, image, max_dist, alpha)

	def draw_heatmap(self, image, alpha = 0.5):

		return util_draw_heatmap(self.heatmap, image, alpha)

	def draw_points_2d(self, image):
		if(not self.poincloudEstimator):
			print("You need to pass the path to the autoencoder model to get the point cloud data")
			return None
		return util_draw_2d(self.points_2d_list, self.boxes_2d_list, 
								  self.axes_2d_list, image, self.label_ids)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':

	model_path = "../models/CenterSnap_sim.onnx"
	poincloud_estimator_path = "../models/CenterSnapAE_sim.onnx"

	# Read rgb image and depth map
	# Download data from the original repository: https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz
	img_path = "../test/scene_2/0073_color.png"
	depth_path = "../test/scene_2/0073_depth.png"
	rgb_img, depth_norm, actual_depth = load_img_NOCS(img_path, depth_path)

	# Initialize the Open3d visualizer
	open3dVisualizer = Open3dVisualizer()

	# Initialize pose estimator
	poseEstimator = CenterSnap(model_path, poincloud_estimator_path, min_conf=0.5)

	# Update pose estimator
	ret = poseEstimator(rgb_img, depth_norm)

	# Draw RGB image with 2d data
	combined_img = poseEstimator.draw_points_2d(rgb_img)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
	cv2.imshow("Output", combined_img)

	# Draw 3D data
	open3dVisualizer(poseEstimator.points_3d_list, poseEstimator.boxes_3d_list, is_image=True)



