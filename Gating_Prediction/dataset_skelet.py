import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import random
import torch

class skeletdataset(data.Dataset):
	def __init__(self, category, data, input_seq_length, output_seq_length, samples, add_noise=False):
		self.category = category
		self.data = data
		self.input_length = input_seq_length
		self.output_length = output_seq_length
		self.total = 0
		self.index = []
		for data in self.data:
			self.total = max(self.total, len(data) * samples)
			self.index.append(len(data))
		self.add_noise = add_noise

	def augment(self, data):
		Range = [0.1, 0.1, 0.1, 0.5, 0.1] * 60

		R = np.random.uniform(size=300)-0.5
		data[-300:] = data[-300:] + R * Range

		return data

	def blur(self, data):
		ed = random.randint(5, 55)

		data[:ed*9] = np.repeat(data[ed*9:(ed+1)*9], ed)

		data[-300:-300+ed*5] = np.repeat(data[-300+ed*5:-300+(ed+1)*5], ed)

		return data

	def __getitem__(self, index):
		idx = index // self.total
		data = self.data[idx]
		index = index % self.total % len(self.data[idx])

		# pre_character, future character, pre camera
		data = data[index]
		character = data[:, :9]
		camera = data[:, -5:]

		index = np.random.randint(len(data)-self.output_length)

		if index >= self.input_length:
			pre_character = character[index-self.input_length:index].flatten()
			pre_camera = camera[index-self.input_length:index].flatten()
		else:
			pre_character = np.concatenate((np.repeat(character[:1], self.input_length-index),
										   character[:index].flatten()))
			pre_camera = np.concatenate((np.repeat(camera[:1], self.input_length-index),
										camera[:index].flatten()))

		if index + self.input_length <= len(data):
			suf_character = character[index:index+self.input_length].flatten()
		else:
			suf_character = np.concatenate((character[index:].flatten(), np.repeat(character[-1:], index+self.input_length-len(data))))

		suf_camera = camera[index:index+self.output_length].flatten()

		global_seq = data
		frame = len(global_seq)
		length = min(400, frame)

		st = random.randint(0, frame-length)
		global_seq = global_seq[st:st+length]

		local_seq = np.concatenate((pre_character, suf_character, pre_camera))
		label = suf_camera

		if (np.random.randint(6) % 2 == 0 and self.add_noise):
			local_seq = self.augment(local_seq)

		if (np.random.randint(6) % 3 == 0 and self.add_noise):
			local_seq = self.blur(local_seq)

		return self.category[idx], global_seq, local_seq, label

	def __len__(self):
		return self.total * len(self.data)