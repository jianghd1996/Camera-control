import os
from agent import agent

class Config(object):
	def __init__(self):
		self.learning_config()
		self.model_config()

	def learning_config(self):
		self.learning_rate = 1e-3
		self.weight_decay = 1e-5
		self.reduce_rate = 0.97
		self.epoch = 300
		self.batch_size = 2048
		self.num_workers = 4
		self.save_freq = 5
		self.eval_freq = 5

	def model_config(self):
		self.channels = [28, 64, 128]
		self.fc_dim = 128
		self.seq_length = 8

		self.pretrain = False
		self.load_name = "latest.pth.tar"
		self.log_dir = "log"
		self.model_path = "model"

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		if not os.path.exists(self.model_path):
			os.mkdir(self.model_path)

def main():
	
	estimation_model = agent(
		train_data_path = [
			"data/fov45_10degree/",
			"data/new_data/",
			"data/new_close_character/",
			"data/fov45_sin_cos/",
			"data/fov45_sin_cos_2/",
			"data/fov45_parallel/",
			"data/fov45_complement/",
			"data/fov45_side/",
			],
		val_data_path = [
			"data/generality/",
			"data/generality_rotation/",
			"data/generality_rotation_2/",
			"data/generality_test/"
			],
		config=Config(),
		)
	estimation_model.learn()


if __name__ == "__main__":
	main()