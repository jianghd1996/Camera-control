import sys
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import Utils as utils
from tqdm import tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm
from dataset_skelet import skeletdataset
from mpl_toolkits.mplot3d import Axes3D
import Net
from Net import Prediction
import os
import random
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class weight_mse(nn.Module):
	def __init__(self):
		super(weight_mse, self).__init__()

	def forward(self, x, y):
		batch_size, length = x.size()
		z = np.array([10] * 5 + [1] * (length - 5))
		z = torch.FloatTensor(np.tile(z, (batch_size, 1))).cuda()
		return torch.mean(z * torch.pow((x - y), 2))


class TrainClock(object):
	def __init__(self):
		self.epoch = 0
		self.step = 0

	def tick(self):
		self.step += 1

	def tock(self):
		self.epoch += 1

	def make_ckpt(self):
		return {
			"epoch": self.epoch,
			"step" 	: self.step
		}

	def load_ckpt(self, ckpt):
		self.epoch = ckpt['epoch']
		self.step = ckpt['step']

def cycle(iterable):
	while True:
		for x in iterable:
			yield x

class Toric_prediction_model(object):
	def __init__(self, train_path, eval_path, track_style, track_shot, config):
		self.train_path   = train_path
		self.eval_path	  = eval_path
		self.track_style = track_style
		self.track_shot  = track_shot
		self.config		 = config
		self.clock		 = TrainClock()
		self.build_model()
		self.load_data()

		self.train_tb = SummaryWriter(os.path.join(config.log_dir, 'train_{}.events'.format(config.num_experts)))
		self.valid_tb = SummaryWriter(os.path.join(config.log_dir, 'valid_{}.events'.format(config.num_experts)))


	def build_model(self):
		print("building model")

		self.model = Prediction(num_experts= self.config.num_experts,
								input_size = self.config.input_size,
								output_size= self.config.output_size,
								hidden_size= self.config.hidden_size)

		if (self.config.pretrain):
			self.load_checkpoint(self.config.load_name)

		self.policies = self.model.parameters()

		self.optimizer = torch.optim.Adam(self.policies, lr=self.config.learning_rate, betas=(0.9, 0.999),
			eps=1e-8, weight_decay=self.config.weight_decay, amsgrad=False)

		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= self.config.reduce_rate)
		self.criterion = weight_mse().cuda()

		self.model = torch.nn.DataParallel(self.model).cuda()

	def load_data(self):
		print("loading data")
		category = []
		data = []

		for path in self.train_path:
			for shot in self.track_shot:
				for i in range(len(self.track_style)):
					category.append(i)
					data.append(np.load
						(os.path.join(path, "{}_{}".format(self.track_style[i], shot)
											  , "toric_prediction_data.npy")))

		self.train_loader = torch.utils.data.DataLoader(
				skeletdataset(category, data, input_seq_length=self.config.input_seq_length,
							  output_seq_length=self.config.output_seq_length,
							  samples = self.config.sample, add_noise=True),
					batch_size=self.config.batch_size,
					shuffle=True,
					num_workers=self.config.num_workers,
					pin_memory=True,
					drop_last=True)

		category = []
		data = []
		for path in self.eval_path:
			for shot in self.track_shot:
				for i in range(len(self.track_style)):
					category.append(i)
					data.append(np.load(
						os.path.join(path, "{}_{}".format(self.track_style[i], shot), "toric_prediction_data.npy")))

		self.val_loader = torch.utils.data.DataLoader(
			skeletdataset(category, data, input_seq_length=self.config.input_seq_length,
						  output_seq_length=self.config.output_seq_length,
						  samples=self.config.sample, add_noise=True),
			batch_size=self.config.batch_size,
			shuffle=False,
			num_workers=self.config.num_workers,
			pin_memory=True,
			drop_last=True)

		self.val_loader = cycle(self.val_loader)

	def learn(self):

		for e in range(self.clock.epoch, self.config.epoch):
			pbar = tqdm(self.train_loader)
			self.model.train()
			for b, data in enumerate(pbar):
				category, global_seq, local_seq, label = data
				global_seq = global_seq.cuda()
				local_seq = local_seq.cuda()
				outputs, losses = self.forward(global_seq, local_seq, label)
				self.update_network(losses)
				self.clock.tick()
				pbar.set_description("EPOCH[{}][{}]".format(e, b))
				pbar.set_postfix(OrderedDict({"loss" : losses.item()}))

				self.record_losses(losses, "train")

				if self.clock.step % self.config.eval_freq == 0:
					self.model.eval()
					data = next(self.val_loader)
					category, global_seq, local_seq, label = data
					global_seq = global_seq.cuda()
					local_seq = local_seq.cuda()
					with torch.no_grad():
						outputs, losses = self.forward(global_seq, local_seq, label)
						self.record_losses(losses, "valid")
					self.model.train()

			self.update_learning_rate()
			self.clock.tock()

			if self.clock.epoch % self.config.save_freq == 0:
				self.save_ckpt(self.clock.epoch)
			self.save_ckpt()

	def forward(self, global_seq, local_seq, label):
		output = self.model(global_seq, local_seq)
		loss = self.criterion(output, label.cuda())
		return output, loss

	def record_losses(self, loss, mode='train'):
		if mode == "train":
			self.train_tb.add_scalar("loss", loss, self.clock.step)
		else:
			self.valid_tb.add_scalar("loss", loss, self.clock.step)

	def update_network(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def update_learning_rate(self):
		self.train_tb.add_scalar("learning_rate", self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
		self.scheduler.step(self.clock.epoch)

	def save_ckpt(self, name=None):
		if name is None:
			save_path = os.path.join(self.config.model_path, "latest.pth.tar")
		else:
			save_path = os.path.join(self.config.model_path, "{}.pth.tar".format(name))

		torch.save({
			"model_state_dict" : self.model.cpu().state_dict(),
			"optimizer_state_dict" : self.optimizer.state_dict(),
			"scheduler_state_dict" : self.scheduler.state_dict(),
			"clock_state_dict" : self.clock.make_ckpt(),
		}, save_path)

		self.model.cuda()

	def load_ckpt(self, name=None):
		if name is None:
			load_path = os.path.joint(self.config.model_path, "latest.pth.tar")
		else:
			load_path = os.path.join(self.config.model_path, "{}.pth.tar".format(name))

		checkpoint = torch.load(load_path)

		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		self.clock.load_ckpt(checkpoint["clock_state_dict"])
