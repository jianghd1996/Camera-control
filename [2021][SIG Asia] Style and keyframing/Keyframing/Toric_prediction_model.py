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
import math as mt
from Net import GateLSTM
import os
import random
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import chain
cudnn.benchmark = True

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
	def __init__(self, train_path, eval_path, config):
		self.train_path   = train_path
		self.eval_path	  = eval_path
		self.config		 = config
		self.clock		 = TrainClock()

		self.train_tb = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
		self.valid_tb = SummaryWriter(os.path.join(config.log_dir, 'valid.events'))

		self.build_model()
		self.load_data()

	def build_model(self):
		print("building model")

		self.model = GateLSTM(
			input_size = self.config.lstm_input_size,
			hidden_size = self.config.lstm_hidden_size,
			output_size = self.config.lstm_output_size,
			gating_input_size = self.config.gating_input_size,
			gating_hidden_size = self.config.gating_hidden_size,
			gating_output_size = self.config.gating_output_size,
		)

		print(self.model)

		self.optimizer = torch.optim.Adam(self.model.parameters(),
										  lr=self.config.learning_rate, betas=(0.9, 0.999),
										  eps=1e-8, weight_decay=self.config.weight_decay, amsgrad=False)

		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= self.config.reduce_rate)
		self.criterion = nn.MSELoss(reduction='none').cuda()

		self.model = self.model.cuda()

		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.reduce_rate)

	def load_data(self):
		print("loading data")

		data = []

		Style = ['direct', 'relative', 'side', 'sin_cos']
		Shot = ['close', 'medium_left', 'medium_mid', 'medium_right', 'far_left', 'far_right']

		for i in range(len(Style)):
			for shot in Shot:
				style = Style[i]
				raw_data = np.load(os.path.join(self.train_path, "{}_track_{}".format(style, shot), "toric_prediction_data.npy"), allow_pickle=True)[()]

				repeat_time = 1
				if i < 2:
					repeat_time = 11

				for l in range(repeat_time):
					for j in range(self.config.down_sample_ratio):
						d = raw_data[:, j::self.config.down_sample_ratio]

						for k in range(len(d)):
							data.append(d[k])

		movie_data_path = "/home/kpl/code/0 Movie/process_movie_data.npy"

		d = np.load(movie_data_path, allow_pickle=True)
		for v in d:
			character = v[:, 5:]
			camera = v[:, :5]
			n = len(character)
			if n < 210:
				continue
			M = np.zeros((n, 2), dtype="float32")
			M[:, 0] = 1
			data.append(np.concatenate((character, M, camera), axis=1))
			M[:, 0] = 0
			M[:, 1] = 1
			data.append(np.concatenate((character, M, camera), axis=1))

		self.train_loader = torch.utils.data.DataLoader(
				skeletdataset(data,
							  gating_length = self.config.gating_length,
							  example_length=self.config.example_length,
							  max_length = self.config.max_length,
							  time_schedule_size = self.config.time_schedule_size),
					batch_size=self.config.batch_size,
					shuffle=True,
					num_workers=self.config.num_workers,
					pin_memory=False,
					drop_last=True)

		data = []

		for i in range(len(Style)):
			for shot in Shot:
				style = Style[i]
				raw_data = \
				np.load(os.path.join(self.eval_path, "{}_track_{}".format(style, shot), "toric_prediction_data.npy"),
						allow_pickle=True)[()]

				for j in range(self.config.down_sample_ratio):
					d = raw_data[:, j::self.config.down_sample_ratio]

					for k in range(len(d)):
						data.append(d[k])

		self.eval_loader = torch.utils.data.DataLoader(
			skeletdataset(data,
						  gating_length=self.config.gating_length,
						  example_length=self.config.example_length,
						  max_length=self.config.max_length,
						  time_schedule_size=self.config.time_schedule_size),
			batch_size=self.config.batch_size,
			shuffle=False,
			num_workers=self.config.num_workers,
			pin_memory=False,
			drop_last=True)

	def visual_latent(self):
		self.load_ckpt(self.model)
		self.model.eval()

		data = []
		latent = []
		label = []

		Style = ['direct', 'relative', 'side', 'sin_cos']
		Shot = ['close', 'medium_left', 'medium_mid', 'medium_right', 'far_left', 'far_right']


		for i in range(len(Style)):
			for shot in Shot:
				style = Style[i]
				raw_data = \
				np.load(os.path.join(self.train_path, "{}_track_{}".format(style, shot), "toric_prediction_data.npy"),
						allow_pickle=True)[()]

				repeat_time = 1
				if i < 2:
					repeat_time = 11

				for l in range(repeat_time):
					d = raw_data[:, ::self.config.down_sample_ratio]

					for k in range(len(d)):
						data.append(d[k][:60])
						label.append(i)

		N = len(data)

		for i in tqdm(range(0, len(data), 2048)):
			l = self.model(torch.tensor(data[i:i+2048]).cuda()).detach().cpu().numpy()[-1]
			for v in l:
				latent.append(v)

		file = open("latent.txt", "w")

		for i in range(len(latent)):
			text = str(label[i]) + " "
			for j in range(4):
				text += str(latent[i][j]) + " "
			file.write(text+"\n")

		file.close()

		# data = []
		#
		# movie_data_path = "/home/kpl/code/0 Movie/process_movie_data.npy"
		#
		# d = np.load(movie_data_path, allow_pickle=True)
		# for v in d[::5]:
		# 	character = v[:, 5:]
		# 	camera = v[:, :5]
		# 	n = len(character)
		# 	if n < 210:
		# 		continue
		# 	M = np.zeros((n, 2), dtype="float32")
		# 	M[:, 0] = 1
		# 	data.append(np.concatenate((character, M, camera), axis=1)[:200])
		# 	label.append(4)
		# 	M[:, 0] = 0
		# 	M[:, 1] = 1
		# 	data.append(np.concatenate((character, M, camera), axis=1)[:200])
		# 	label.append(4)
		#
		# for i in tqdm(range(0, len(data), 2048)):
		# 	l = self.model(torch.tensor(data[i:i+2048]).cuda()).detach().cpu().numpy()[-1]
		# 	for v in l:
		# 		latent.append(v)

		if not os.path.exists("visual"):
			os.mkdir("visual")

		pca = PCA(n_components=2)
		x_pca = pca.fit_transform(latent)
		plt.scatter(x_pca[:N, 0], x_pca[:N, 1], c=label[:N])
		plt.savefig("syn_pca.png")
		plt.close()

		plt.scatter(x_pca[:, 0], x_pca[:, 1], c=label)
		plt.savefig("pca.png")
		plt.close()

		tsne = TSNE(n_components=2)
		x_tsne = tsne.fit_transform(latent)
		plt.scatter(x_tsne[:N, 0], x_tsne[:N, 1], c=label[:N])
		plt.savefig("syn_tsne.png")
		plt.close()

		plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=label)
		plt.savefig("tsne.png")
		plt.close()

	def learn_LSTM(self):
		for e in range(self.clock.epoch, self.config.epoch):
			self.teacher_forcing_ratio = 0

			pbar = tqdm(self.train_loader)
			self.model.train()
			for b, data in enumerate(pbar):
				gating_seq, character, camera, Mask, Schedule, Target, start_traj = data
				gating_seq = gating_seq.cuda()
				character = character.cuda()
				camera = camera.cuda()
				Mask = Mask.cuda()
				Schedule = Schedule.cuda()
				Target = Target.cuda()
				start_traj = start_traj.cuda()
				outputs, rec_loss, keyframe_loss = self.forward(gating_seq, character, camera, Mask, Schedule, Target, start_traj)
				losses = rec_loss + keyframe_loss
				self.update_network(self.optimizer, losses)

				pbar.set_description("EPOCH[{}][{}]".format(e, b))
				pbar.set_postfix(OrderedDict({"r_loss" : rec_loss.item(),
											  "k_loss": keyframe_loss.item(),
											  }))

				self.clock.tick()
				self.record_losses(rec_loss, "train", "rec_loss")
				self.record_losses(keyframe_loss, "train", "keyframe_loss")

			self.update_learning_rate(self.scheduler, e)
			self.clock.tock()

			if self.clock.epoch % self.config.save_freq == 0:
				self.save_ckpt(self.model)

			if e % self.config.eval_freq == 0:
				pbar = tqdm(self.eval_loader)
				self.model.eval()
				tot_loss = []
				for b, data in enumerate(pbar):
					gating_seq, character, camera, Mask, Schedule, Target, start_traj = data
					gating_seq = gating_seq.cuda()
					character = character.cuda()
					camera = camera.cuda()
					Mask = Mask.cuda()
					Schedule = Schedule.cuda()
					Target = Target.cuda()
					start_traj = start_traj.cuda()
					outputs, rec_loss, keyframe_loss = self.forward(gating_seq, character, camera,
																								 Mask, Schedule, Target, start_traj)
					loss = np.mean(abs(outputs - camera).detach().cpu().numpy(), axis=1)
					for v in loss:
						tot_loss.append(v)

				print(outputs[0][-1], camera[0][-1], Target[0][-1][-5:])

				tot_loss = np.array(tot_loss)
				name = ['pA', 'pB', 'pY', 'theta', 'phi']
				print("Evaluating Epoch {}".format(e))
				for i in range(len(name)):
					print("{} loss mean {} var {}".format(name[i], tot_loss[:, i].mean(), tot_loss[:, i].var()))

	def forward(self, gating_seq, character, camera, Mask, Schedule, Target, start_traj):
		batch_size = len(character)

		out_camera = [camera[:, 0]]

		latent_vec_seq = self.model(gating_seq)
		latent_vec = latent_vec_seq[self.config.example_length-1]

		hidden = self.model.initHidden(torch.cat([latent_vec, start_traj], dim=1)).cuda()

		hidden_energy_loss = None
		for i in range(self.config.max_length):
			lstm_input = torch.cat([latent_vec, character[:, i], out_camera[-1], Schedule[:, i], Target[:, i]], dim=1)

			output, hidden = self.model(lstm_input, hidden)

			out_camera.append(output)

		out_camera = torch.stack(out_camera, dim=1)

		loss = torch.sum(self.criterion(out_camera[:, 1:], camera[:, 1:]), dim=2)
		keyframe_loss = self.criterion(out_camera[:, 1:], camera[:, 1:]) * Mask.repeat(1, 1, 5)

		return out_camera, loss.mean(), keyframe_loss.sum() / Mask.sum()

	def build_time_schedule(self):
		tss = self.config.time_schedule_size
		self.schedule = np.zeros((self.config.max_length, tss), dtype="float32")
		self.schedule_scale = tss
		basis = 10000
		for tta in range(self.config.max_length):
			for i in range(tss // 2):
				self.schedule[tta][2 * i] = mt.sin(1.0 * tta / mt.pow(basis, 2.0 * i / tss))
				self.schedule[tta][2 * i + 1] = mt.cos(1.0 * tta / mt.pow(basis, 2.0 * i / tss))

		nz = np.zeros((tss, self.config.max_length, 3), dtype="uint8")
		for tta in range(self.config.max_length):
			for i in range(tss):
				if self.schedule[tta][i] >= 0:
					nz[i][tta] = np.array([140, 140, 30]) + np.array([-110, 90, 210]) * self.schedule[tta][i]
				else:
					nz[i][tta] = np.array([140, 140, 30]) + np.array([-60, -140, 30]) * self.schedule[tta][i]

	def freeze(self, model):
		for name, module in model._modules.items():
			for p in module.parameters():
				p.requires_grad = False

	def record_losses(self, loss, mode='train', name="loss"):
		if mode == "train":
			self.train_tb.add_scalar(name, loss, self.clock.step)
		else:
			self.valid_tb.add_scalar(name, loss, self.clock.step)

	def update_network(self, optimizer, loss):
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	def update_learning_rate(self, scheduler, epoch):
		scheduler.step(epoch)

	def save_ckpt(self, model, cate="", name=None):
		if name is None:
			save_path = os.path.join(self.config.model_path, cate+"latest.pth.tar")
		else:
			save_path = os.path.join(self.config.model_path, cate+"{}.pth.tar".format(name))

		torch.save({
			"model_state_dict" : model.cpu().state_dict(),
			"optimizer_state_dict" : self.optimizer.state_dict(),
			"scheduler_state_dict" : self.scheduler.state_dict(),
			"clock_state_dict" : self.clock.make_ckpt(),
		}, save_path)

		model.cuda()

	def load_ckpt(self, model, cate="", name=None, only_model=False):
		if name is None:
			load_path = os.path.join(self.config.model_path, cate + "latest.pth.tar")
		else:
			load_path = os.path.join(self.config.model_path, cate + "{}.pth.tar".format(name))
		print("load checkpoint from {}".format(load_path))

		checkpoint = torch.load(load_path)

		if only_model:
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			model.load_state_dict(checkpoint["model_state_dict"])
			self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
			self.clock.load_ckpt(checkpoint["clock_state_dict"])