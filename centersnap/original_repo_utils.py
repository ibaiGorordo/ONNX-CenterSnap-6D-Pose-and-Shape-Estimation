import numpy as np
import cv2

from skimage.feature import peak_local_max

NOCS_CAMERA_MAT = np.array([[591.0125 ,   0.     , 322.525  ,   0.     ],
       						[  0.     , 590.16775, 244.11084,   0.     ],
       						[  0.     ,   0.     ,   1.     ,   0.     ],
       						[  0.     ,   0.     ,   0.     ,   1.     ]])

open_3d_lines = [
        [0, 1],
        [7,3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/nocs_utils.py#L7
def load_depth(depth_path):
	""" Load depth image from img_path. """
	# depth_path = depth_path + '_depth.png'
	# print("depth_path", depth_path)
	depth = cv2.imread(depth_path, -1)
	if len(depth.shape) == 3:
		# This is encoded depth image, let's convert
		# NOTE: RGB is actually BGR in opencv
		depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
		depth16 = np.where(depth16==32001, 0, depth16)
		depth16 = depth16.astype(np.uint16)
	elif len(depth.shape) == 2 and depth.dtype == 'uint16':
		depth16 = depth
	else:
		assert False, '[ Error ]: Unsupported depth type.'
	return depth16

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/nocs_utils.py#L24
def load_img_NOCS(rgm_img_path, depth_path):

	left_img = cv2.imread(rgm_img_path)
	depth = load_depth(depth_path)
	depth_norm = np.array(depth, dtype=np.float32)/255.0

	return left_img, depth_norm, depth

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/pose_outputs.py#L121
def find_nearest(peaks,value):
	newList = np.linalg.norm(peaks-value, axis=1)
	return peaks[np.argsort(newList)]

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/pose_outputs.py#L126
def extract_peaks_from_centroid_sorted(centroid_heatmap,min_confidence=0.15, min_distance=10):
	peaks = peak_local_max(centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence)
	peaks = find_nearest(peaks,[0,0])
	return peaks

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/pose_outputs.py#L131
def extract_latent_emb_from_peaks(heatmap_output, peaks, latent_emb_output, scale_factor=8):
	assert peaks.shape[1] == 2
	latent_embeddings = []
	indices = []
	scores = []
	for ii in range(peaks.shape[0]):
		index = np.zeros([2])
		index[0] = int(peaks[ii, 0] / scale_factor)
		index[1] = int(peaks[ii, 1] / scale_factor)
		index = index.astype(np.int)
		latent_emb = latent_emb_output[index[0], index[1], :]
		latent_embeddings.append(latent_emb)
		indices.append(index*scale_factor)
		scores.append(heatmap_output[peaks[ii, 0], peaks[ii, 1]])
	return latent_embeddings, indices, scores

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/pose_outputs.py#L147
def extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=8):
	assert peaks.shape[1] == 2
	abs_poses = []
	scales = []
	for ii in range(peaks.shape[0]):
		index = np.zeros([2])
		index[0] = int(peaks[ii, 0] / scale_factor)
		index[1] = int(peaks[ii, 1] / scale_factor)
		index = index.astype(np.int)

		abs_pose_values = abs_pose_output[index[0], index[1],:]
		rotation_matrix = np.array([[abs_pose_values[0], abs_pose_values[1], abs_pose_values[2]],
																[abs_pose_values[3], abs_pose_values[4], abs_pose_values[5]],
																[abs_pose_values[6], abs_pose_values[7], abs_pose_values[8]]])
		translation_vector = np.array([abs_pose_values[9], abs_pose_values[10], abs_pose_values[11]])
		
		transformation_mat = np.eye(4)
		transformation_mat[:3,:3] = rotation_matrix
		transformation_mat[:3,3] = translation_vector

		scale = abs_pose_values[12]
		scale_matrix = np.eye(4)
		scale_mat = scale*np.eye(3, dtype=float)
		scale_matrix[0:3, 0:3] = scale_mat
		scales.append(scale_matrix)

		abs_poses.append((transformation_mat, scale_matrix))
	return abs_poses

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/pose_outputs.py#L177
def draw_peaks(centroid_target, peaks):
	centroid_target = np.clip(centroid_target, 0.0, 1.0) * 255.0
	heatmap_img = cv2.applyColorMap(centroid_target.astype(np.uint8), cv2.COLORMAP_JET)
	for ii in range(peaks.shape[0]):
		point = (int(peaks[ii, 1]), int(peaks[ii, 0]))
		heatmap_img = cv2.putText(heatmap_img,str(ii), 
		point, 
		cv2.FONT_HERSHEY_SIMPLEX, 
		1,
		(255,255,255),
		2)
		cv2.line(heatmap_img, point, (0,0), (0, 255, 0), thickness=3, lineType=8)
	return heatmap_img

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/abs_pose_outputs.py#L120
def compute_point_cloud_embeddings(heatmap_output, latent_emb_output, min_confidence):
	peaks = extract_peaks_from_centroid_sorted(np.copy(heatmap_output), min_confidence)
	#peaks_image = None
	peaks_image = draw_peaks(np.copy(heatmap_output), np.copy(peaks))
	latent_embs, indices, scores = extract_latent_emb_from_peaks(
			np.copy(heatmap_output), np.copy(peaks), np.copy(latent_emb_output)
	)
	return latent_embs, peaks, peaks_image,scores, indices

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/net/post_processing/abs_pose_outputs.py#L129
def compute_pointclouds_and_poses(
		heatmap_output,
		latent_emb_output,
		abs_pose_output,
		min_confidence
):
	latent_embeddings , peaks, img,scores, indices = compute_point_cloud_embeddings(np.copy(heatmap_output), np.copy(latent_emb_output), min_confidence)

	abs_pose_outputs = extract_abs_pose_from_peaks(np.copy(peaks), abs_pose_output)
	return latent_embeddings, abs_pose_outputs, img, scores, indices

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/camera.py#L106
def convert_points_to_homopoints(points):
	"""Project 3d points (3xN) to 4d homogenous points (4xN)"""
	assert len(points.shape) == 2
	assert points.shape[0] == 3
	points_4d = np.concatenate([
	  points,
	  np.ones((1, points.shape[1])),
	], axis=0)
	assert points_4d.shape[1] == points.shape[1]
	assert points_4d.shape[0] == 4
	return points_4d

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/simnet/lib/camera.py#L119
def convert_homopoints_to_points(points_4d):
	"""Project 4d homogenous points (4xN) to 3d points (3xN)"""
	assert len(points_4d.shape) == 2
	assert points_4d.shape[0] == 4
	points_3d = points_4d[:3, :] / points_4d[3:4, :]
	assert points_3d.shape[1] == points_3d.shape[1]
	assert points_3d.shape[0] == 3
	return points_3d

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/transform_utils.py#L29
def project(K, p_3d):
	projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
	p_2d = np.dot(K, p_3d)
	projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
	projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
	return projections_2d

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/transform_utils.py#L118
def calculate_2d_projections(coordinates_3d, intrinsics):
	"""
	Input: 
		coordinates: [3, N]
		intrinsics: [3, 3]
	Return 
		projected_coordinates: [N, 2]
	"""
	projected_coordinates = intrinsics @ coordinates_3d
	projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
	projected_coordinates = projected_coordinates.transpose()
	projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

	return projected_coordinates

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/transform_utils.py#L65
def get_3d_bbox(size, shift=0):
	"""
	Args:
		size: [3] or scalar
		shift: [3] or scalar
	Returns:
		bbox_3d: [3, N]
	"""
	bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
					[+size[0] / 2, +size[1] / 2, -size[2] / 2],
					[-size[0] / 2, +size[1] / 2, +size[2] / 2],
					[-size[0] / 2, +size[1] / 2, -size[2] / 2],
					[+size[0] / 2, -size[1] / 2, +size[2] / 2],
					[+size[0] / 2, -size[1] / 2, -size[2] / 2],
					[-size[0] / 2, -size[1] / 2, +size[2] / 2],
					[-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
	return bbox_3d

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/transform_utils.py#L102
def transform_coordinates_3d(coordinates, RT):
	"""
	Input: 
		coordinates: [3, N]
		RT: [4, 4]
	Return 
		new_coordinates: [3, N]
	"""
	assert coordinates.shape[0] == 3
	coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
	new_coordinates = RT @ coordinates
	new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
	return new_coordinates

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/transform_utils.py#L37
def get_pointclouds_3d(pose, pc):

	pc_homopoints = convert_points_to_homopoints(pc.T)
	morphed_pc_homopoints = pose[0] @ (pose[1] @ pc_homopoints)
	morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

	pc_hp = convert_points_to_homopoints(pc.T)
	scaled_homopoints = (pose[1] @ pc_hp)
	scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
	size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
	box = get_3d_bbox(size)
	unit_box_homopoints = convert_points_to_homopoints(box.T)
	morphed_box_homopoints = pose[0] @ unit_box_homopoints
	morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T

	xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
	sRT = pose[0] @ pose[1]
	transformed_axes = transform_coordinates_3d(xyz_axis, sRT)

	return morphed_pc_homopoints, morphed_box_points, transformed_axes, size

def get_pointclouds_2d(pointcloud_3d, box_3d, axes_3d, camera_mat):

	points_3d_hp = convert_points_to_homopoints(pointcloud_3d.T)
	points_2d_mesh = project(camera_mat, points_3d_hp)
	points_2d_mesh = points_2d_mesh.T

	box_3d_hp = convert_points_to_homopoints(np.array(box_3d).T)
	box_2d = project(camera_mat, box_3d_hp)
	box_2d = box_2d.T

	projected_axes = calculate_2d_projections(axes_3d, camera_mat[:3,:3])

	return points_2d_mesh, box_2d, projected_axes

# Ref: https://github.com/zubair-irshad/CenterSnap/blob/5422258475c30c37807566c60996f4d8b3a810e7/utils/viz_utils.py#L238
def draw_bboxes(img, img_pts, axes, color):

	img_pts = np.int32(img_pts).reshape(-1, 2)
	# draw ground layer in darker color
	
	# color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
	color_ground = (int(color[0]), int(color[1]), int(color[2]))
	
	for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
		img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
	# draw pillars in minor darker color
	# color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
	color_pillar = (int(color[0]), int(color[1]), int(color[2]))
	for i, j in zip(range(4), range(4, 8)):
		img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
	# draw top layer in original color
	for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
		img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

	# draw axes
	img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 2)
	img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 2)
	img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4) ## y last

	return img


