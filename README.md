# ONNX-CenterSnap-6D-Pose-and-Shape-Estimation
 Python scripts for performing 6D pose estimation and shape reconstruction using the CenterSnap model in ONNX
 
![ONNX-CenterSnap-6D-Pose-and-Shape-Estimation](https://user-images.githubusercontent.com/43162939/159124756-6c562b36-f901-4669-b003-f7dec714b684.png)

# Important
 - The original model has not been officially released, therefore, there might be changes to the official model later on.
 - The examples seem to not properly work when using a camera other than the one in the original dataset. This is probably due to an implementation mistake on this repository, if you find the issue, please submit an issue or PR.
 - The model only works with the following objects (the objects avaialble in the training dataset): bottle, bowl, camera, can, laptop, and mug.

# Requirements

 * Check the requirements.txt file.
 * Additionally depthai library is necessary for testing with OAK-D boards. Check the example below for how to install it.
 * Similarly, you will need to commit the pyKinectAzure repository to run the example with the Azure Kinect. Check the example below for how to install it.
 
# Installation
```
pip install -r requirements.txt
```

# ONNX model

Download the models from [here](https://drive.google.com/file/d/1bRIBWPWwqYg7sGglqF71XmJsfiHvLR1L/view?usp=sharing) and [here](https://drive.google.com/file/d/1UVmhwJV605T_iJ90QKTN79kXMC40EydV/view?usp=sharing), and place them in the [models](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/tree/main/models) folder. For converting the original Pytorch model to ONNX, check the following branch: https://github.com/ibaiGorordo/CenterSnap/tree/convert_onnx

# How to use

 The model returns multiple outputs (segmentation map, heat map, 3D position...), here is a short explanation on how to run these examples. For the image examples, you will need to download the data from the original source: https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz

 * **Image Segmentation map**:

 ![CenterSnap Semantic map](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/blob/main/doc/img/segmentation.png)
 
 ```
 python image_draw_semantic_map.py
 ```

 * **Image heatmap**:

 ![CenterSnap heatmap](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/blob/main/doc/img/heatmap.png)
 
 ```
 python image_draw_heatmap.py
 ```
 
 * **Image depth map (Same as input)**:

 ![CenterSnap depth map](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/blob/main/doc/img/depthmap.png)
 
 ```
 python image_draw_depth.py
 ```
 
  * **Image projected 3D pose**:

 ![CenterSnap projected 3d pose](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/blob/main/doc/img/pose2d.png)
 
 ```
 python image_draw_pose2d.py
 ```

 * **Image 3D pose**:

 ![CenterSnap 3d pose](https://github.com/ibaiGorordo/ONNX-CenterSnap-6D-Pose-and-Shape-Estimation/blob/main/doc/img/pose3d.png)
 
 ```
 python image_draw_pose3d.py
 ```

 * **OAK-D 3D pose**:

https://user-images.githubusercontent.com/43162939/159125339-884517ef-7796-4abc-8e09-e8ed1ad7b43c.mp4

   First, install the depthai library: `pip install depthai`
 
 ```
 python depthai_draw_pose3d.py
 ```

 * **Azure Kinect 3D pose**:

   - First, install the Azure Kinect SDK: https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download
   - Clone the pyKinectAzure repository inside this repository: `git clone https://github.com/ibaiGorordo/pyKinectAzure.git`
 
 ```
 python kinect_draw_pose3d.py
 ```

# References
- **CenterSnap**: https://github.com/zubair-irshad/CenterSnap
- **Original paper**: https://arxiv.org/abs/2203.01929
- **Modified CenterSnap for conversion**: https://github.com/ibaiGorordo/CenterSnap/tree/convert_onnx
- **Depthai library**: https://github.com/luxonis/depthai-python
- **pyKinectAzure**: https://github.com/ibaiGorordo/pyKinectAzure

