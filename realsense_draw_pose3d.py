import cv2
import numpy as np
import pyrealsense2 as rs

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS, Open3dVisualizer


REALSENSE_MAT_640 = np.array([[428.907  ,   0.     , 321.383  ,   0.     ],
                              [  0.     , 428.611  , 241.602  ,   0.     ],
                              [  0.     ,   0.     ,   1.     ,   0.     ],
                              [  0.     ,   0.     ,   0.     ,   1.     ]])


def initialize_device():

    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get stream profile and camera intrinsics
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    # print(color_intrinsics)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale

if __name__ == '__main__':
    
    model_path = "models/CenterSnap_sim.onnx"
    poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"
    max_dist = 2.0

    # Initialize pose estimator with autoencoder
    poseEstimator = CenterSnap(model_path, poincloud_estimator_path, min_conf=0.6, camera_mat=REALSENSE_MAT_640)

    # Create REALSENSE  pipeline
    pipeline, align, depth_scale = initialize_device()

    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920*2,1080))
    
    # Initialize the Open3d visualizer
    open3dVisualizer = Open3dVisualizer()

    cv2.namedWindow('Projected Pose',cv2.WINDOW_NORMAL)
    while True:

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_map = np.asanyarray(aligned_depth_frame.get_data())*depth_scale*1000
        depth_map[depth_map>max_dist*1000] = max_dist*1000
        rgb_img = np.asanyarray(color_frame.get_data())

        # Update pose estimator
        ret = poseEstimator(rgb_img, depth_map/255.0)

        if ret:

            # Draw RGB image with 2d data
            combined_img = poseEstimator.draw_points_2d(rgb_img)

            # Draw 3D data
            open3dVisualizer(poseEstimator.points_3d_list, poseEstimator.boxes_3d_list)
        else:
            combined_img = rgb_img

        combined_img = cv2.resize(combined_img, (1920, 1080))  
        
        # Convert Open3D map to image
        o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
        o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (1920, 1080))  

        combined_img = cv2.hconcat([combined_img, o3d_screenshot_mat])
        # out.write(combined_img)
        cv2.imshow("Projected Pose", combined_img)

    # out.release()
    pipeline.stop()