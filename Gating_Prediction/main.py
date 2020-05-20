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
		self.epoch = 300
		self.batch_size = 256
		self.num_workers = 8
		self.sample = 10

	def model_config(self):
		self.input_seq_length = 60
		self.output_seq_length = 30
		self.num_experts = 9

		self.input_size = self.input_seq_length * (9 + 5 + 9)
		self.output_size = self.output_seq_length * 5
		self.hidden_size = 512

		self.pretrain = False
		self.load_name = "latest.pth.tar"
		self.save_freq = 5
		self.eval_freq = 5
		self.log_dir = "log"
		self.model_path = "model_{}".format(self.num_experts)

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		if not os.path.exists(self.model_path):
			os.mkdir(self.model_path)

def main():
	prediction_model = Toric_prediction_model(
		train_path=["../prediction_feature/"],
		eval_path=["../prediction_evaluation_feature/"],
		track_style=['direct_track', "relative_track", "sin_cos_track", "side_track"],
		track_shot=["close", "far_left", "far_right", "medium_left", "medium_mid", "medium_right"],
		config = Config(),
		)
	prediction_model.build_model()
	prediction_model.learn()

if __name__ == "__main__":
	main()