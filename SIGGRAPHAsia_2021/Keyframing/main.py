import numpy as np
import os
from Toric_prediction_model import Toric_prediction_model

class Config(object):
	def __init__(self):
		self.learning_config()
		self.model_config()

	def learning_config(self):
		self.learning_rate = 1e-3
		self.weight_decay = 1e-5
		self.reduce_rate = 0.97
		self.epoch = 200
		self.batch_size = 1024
		self.num_workers = 8
		self.max_length = 200
		self.down_sample_ratio = 5

	def model_config(self):
		# input : current character, last camera, time_schedule_size, style_vector, target character, target camera
		# output : current camera
		self.example_length = 60

		self.gating_input_size = 9 + 5
		self.gating_hidden_size = 512
		self.gating_output_size = 4
		self.gating_length = 120

		self.time_schedule_size = 128

		self.lstm_input_size = self.gating_output_size + 9 + 5+ 9 + 5 + self.time_schedule_size
		self.lstm_hidden_size = 256
		self.lstm_output_size = 5

		self.pretrain = False
		self.load_name = "latest"
		self.save_freq = 20
		self.eval_freq = 1
		self.log_dir = "log"
		self.model_path = "model"

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		if not os.path.exists(self.model_path):
			os.mkdir(self.model_path)

def main():
	prediction_model = Toric_prediction_model(
		train_path="../prediction_feature/",
		eval_path="../prediction_evaluation_feature/",
		config = Config(),
		)
	# prediction_model.learn_LSTM()
	prediction_model.visual_latent()
	# prediction_model.visual_hidden()

if __name__ == "__main__":
	main()