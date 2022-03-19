import cv2
import numpy as np
import sys

import depthai as dai

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS, Open3dVisualizer

# Ref: https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/rgb_depth_aligned.py
def create_pipeline():

    fps = 10
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

    # Create pipeline
    pipeline = dai.Pipeline()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    depthOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    depthOut.setStreamName("depth")
    queueNames.append("depth")

    #Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(fps)
    camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    camRgb.initialControl.setManualFocus(130)

    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    return pipeline

DEPTHAI_MAT_720 = np.array([[1492.486328125   ,   0.     , 958.041015625   ,   0.     ],
       					   [  0.     , 1490.5980224609375   , 548.6614379882812   ,   0.     ],
       					   [  0.     ,   0.     ,   1.     ,   0.     ],
       					   [  0.     ,   0.     ,   0.     ,   1.     ]])

model_path = "models/CenterSnap_sim.onnx"
poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"
max_dist = 2.0

# Initialize pose estimator with autoencoder
poseEstimator = CenterSnap(model_path, poincloud_estimator_path, min_conf=0.4, camera_mat=DEPTHAI_MAT_720)

# Create dephtai pipeline
pipeline = create_pipeline()

# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920*2,1080))

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    
    rgb_img = None
    depth_map = None

    # Initialize the Open3d visualizer
    open3dVisualizer = Open3dVisualizer()

    cv2.namedWindow('Projected Pose',cv2.WINDOW_NORMAL)
    while True:

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break

        # Get capture
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["depth"] = None

        queueEvents = device.getQueueEvents(("rgb", "depth"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            rgb_img = latestPacket["rgb"].getCvFrame()

        if latestPacket["depth"] is not None:
            depth_map = latestPacket["depth"].getFrame()
            depth_map = np.ascontiguousarray(depth_map)
            depth_map[depth_map>max_dist*1000] = max_dist*1000

        # Blend when both received
        if rgb_img is None or depth_map is None:
            continue

        # # Update pose estimator
        ret = poseEstimator(rgb_img, depth_map/255.0)

        rgb_img = cv2.resize(rgb_img, (1920, 1080))  
        if ret:

            # Draw RGB image with 2d data
            combined_img = poseEstimator.draw_points_2d(rgb_img)

            # Draw 3D data
            open3dVisualizer(poseEstimator.points_3d_list, poseEstimator.boxes_3d_list)
        else:
            combined_img = rgb_img
        
        # Convert Open3D map to image
        o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
        o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (1920, 1080))  

        combined_img = cv2.hconcat([combined_img, o3d_screenshot_mat])
        # out.write(combined_img)

        cv2.namedWindow("Projected Pose", cv2.WINDOW_NORMAL)
        cv2.imshow("Projected Pose", combined_img)

        rgb_img = None
        frameDisp = None

# out.release()


