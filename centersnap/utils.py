import numpy as np
import cv2
import open3d as o3d

from .original_repo_utils import *

np.random.seed(3)

MAX_CLASS_NUM = 100 # In the original model there are only 7 classes 
segmenation_colors = np.random.randint(0, 255, (MAX_CLASS_NUM, 3)).astype("uint8")

def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = segmenation_colors[seg_map]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img

def util_draw_depth(depth_map, image, max_depth = 2, alpha = 0.5):

	# Normalize estimated depth to color it
	if max_depth:
		min_depth = 0
		depth_map = depth_map/1000 # Convert to meters
	else:
		min_depth = depth_map.min()
		max_depth = depth_map.max()

	norm_depth_map = 255*(depth_map-min_depth)/(max_depth-min_depth)
	norm_depth_map[norm_depth_map < 0] =0
	norm_depth_map[norm_depth_map >= 255] = 255

	# Normalize and color the image
	color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_PLASMA )

	# Resize to match the image shape
	color_depth = cv2.resize(color_depth, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_depth))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_depth, (1-alpha),0)

	return combined_img

def util_draw_heatmap(heatmap, image, alpha = 0.5):

	# Normalize and color the image
	color_heatmap = cv2.applyColorMap(cv2.convertScaleAbs(heatmap*255,1), cv2.COLORMAP_JET)

	# Resize to match the image shape
	color_heatmap = cv2.resize(color_heatmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_heatmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_heatmap, (1-alpha),0)

	return combined_img

def util_draw_points2d(points_2d_list, image, label_ids):

	# Normalize and color the image
	for i, points_2d in enumerate(points_2d_list):

		color = (int(segmenation_colors[label_ids[i]][0]), 
				int(segmenation_colors[label_ids[i]][1]), 
				int(segmenation_colors[label_ids[i]][2]))

		for point in points_2d.astype(int):

			cv2.circle(image, (int(point[0]),int(point[1])), 1, color, -1)

	return image

def util_draw_pose2d(boxes_2d_list, axes_2d_list, image, label_ids):

	# Normalize and color the image
	for i, (box, axis) in enumerate(zip(boxes_2d_list, axes_2d_list)):

		color = (int(segmenation_colors[label_ids[i]][0]*0.5), 
				int(segmenation_colors[label_ids[i]][1]*0.5), 
				int(segmenation_colors[label_ids[i]][2]*0.5))

		image = draw_bboxes(image, box, axis, color)

	return image

def util_draw_2d(points_2d_list, boxes_2d_list, axes_2d_list, image, label_ids):

	image = util_draw_points2d(points_2d_list, image, label_ids)
	return util_draw_pose2d(boxes_2d_list, axes_2d_list, image, label_ids)

class Open3dVisualizer():

	def __init__(self):

		self.point_cloud = o3d.geometry.PointCloud()
		self.boxes = o3d.geometry.LineSet()
		self.o3d_started = False

		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window()

	def __call__(self, points_3d_list, boxes_3d_list, is_image = False):

		self.update(points_3d_list, boxes_3d_list, is_image)

	def update(self, points_3d_list, boxes_3d_list, is_image = False):

		# Process points
		all_points, all_boxes, all_lines = Open3dVisualizer.process_data(points_3d_list, boxes_3d_list)

		# Add values to vectors
		self.point_cloud.points = o3d.utility.Vector3dVector(all_points)
		self.boxes.points = o3d.utility.Vector3dVector(all_boxes)
		self.boxes.lines = o3d.utility.Vector2iVector(all_lines)

		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.point_cloud)
			self.vis.add_geometry(self.boxes)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.point_cloud)
			self.vis.update_geometry(self.boxes)

		self.vis.poll_events()
		self.vis.update_renderer()

	@staticmethod
	def process_data(points_3d_list, boxes_3d_list):

		all_points = points_3d_list[0]
		all_boxes = boxes_3d_list[0]
		all_lines = np.array(open_3d_lines)
		box_count = 0
		for points_3d, box_3d in zip(points_3d_list[1:], boxes_3d_list[1:]):
			box_count += 1
			all_points = np.vstack((all_points, points_3d))
			all_boxes = np.vstack((all_boxes, box_3d))
			all_lines = np.vstack((all_lines, np.array(open_3d_lines)+8*box_count))

		# Fix axis to match open3d
		all_points = -all_points[:,[0,1,2]]
		all_boxes = -all_boxes[:,[0,1,2]]
		all_points[:,0] = -all_points[:,0]
		all_boxes[:,0] = -all_boxes[:,0]
		
		return all_points, all_boxes, all_lines
			


	

	

	
