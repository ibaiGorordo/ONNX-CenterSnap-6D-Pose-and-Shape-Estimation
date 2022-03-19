import cv2
import onnxruntime
import numpy as np

# Estimates the object pointcloud from the embedding (1x128)
class CenterSnapAE():

	def __init__(self, model_path):

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, embeding):
		return self.estimate_points(embeding)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_points(self, embeding):

		input_tensor = self.prepare_input(embeding)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.point_cloud = self.process_output(outputs)

		return self.point_cloud

	def prepare_input(self, embeding):

		return np.expand_dims(embeding, axis=0).astype(np.float32)

	def inference(self, input_tensor):

		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

		# print(time.time() - start)
		return outputs

	def process_output(self, output): 

		_, point_cloud = output

		# Process outputs
		return point_cloud[0]

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	model_path = "../models/CenterSnapAE_sim.onnx"

	test_input = np.array([ 0.01383636,  0.08519881, -0.2568576 , -0.06929949,  0.16245446,
       -0.19117895, -0.08131296, -0.14462729, -0.19391498,  0.15482728,
        0.13780275,  0.20859261, -0.06526352,  0.09346385, -0.13005106,
        0.15697843,  0.06575619,  0.04343044,  0.02837731,  0.12083598,
       -0.15285742,  0.21383464, -0.00493477, -0.06542912,  0.12784249,
        0.04896393, -0.2392243 , -0.06248051,  0.01151521, -0.09024532,
        0.19521667,  0.02702888,  0.09669966, -0.10894367, -0.02628613,
        0.04001294,  0.20138043,  0.02759702,  0.13812885,  0.11197274,
       -0.16526113,  0.20321588,  0.07029293, -0.12045169, -0.14619656,
        0.05837455, -0.11362121,  0.13326073, -0.02754317, -0.0587311 ,
       -0.14827383,  0.1470561 ,  0.01239088,  0.13250925, -0.12047467,
       -0.15799877,  0.06379244, -0.11408932,  0.04498161, -0.06124537,
        0.04647233, -0.22129023,  0.0987699 ,  0.25380003,  0.1124868 ,
        0.03985383,  0.09265739,  0.08266334,  0.09266652, -0.00779468,
        0.20741701, -0.0561517 ,  0.18054232,  0.04004669, -0.12903439,
        0.03250518, -0.01802721,  0.07982591, -0.10568285, -0.21289922,
        0.03140058, -0.21204248, -0.00271844, -0.07429365, -0.16981226,
        0.10670336, -0.08002263,  0.15605289, -0.13053742, -0.13143297,
       -0.23060118, -0.09795808,  0.09555136, -0.00459275, -0.05467268,
        0.06259942,  0.03536284, -0.07436401,  0.03872812,  0.10186   ,
        0.18344708,  0.1885262 ,  0.15772581,  0.1285103 , -0.06618582,
       -0.07889589, -0.07869944,  0.1770217 ,  0.08011568,  0.17497912,
        0.2022954 ,  0.0752039 , -0.05985646, -0.01378733,  0.0329463 ,
        0.03606942, -0.04599845,  0.01062884,  0.20250002, -0.1304602 ,
       -0.10151692,  0.08815864, -0.20215893, -0.20922695, -0.12711483,
       -0.05051517,  0.07056645, -0.03954468], dtype=np.float32)

	pointcloudEstimator = CenterSnapAE(model_path)

	estimated_pointcloud = pointcloudEstimator(test_input)

	# Plot pointcloud, it should show a laptop
	from matplotlib import pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(estimated_pointcloud[:,0],estimated_pointcloud[:,2],estimated_pointcloud[:,1])
	plt.show()
	

