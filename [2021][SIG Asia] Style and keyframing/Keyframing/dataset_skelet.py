import math as mt

import numpy as np
import torch.utils.data as data

class skeletdataset(data.Dataset):
	def __init__(self, data, gating_length, example_length, max_length, time_schedule_size):
		self.data = data
		self.gating_length = gating_length
		self.example_length = example_length
		self.max_length = max_length
		self.tss = time_schedule_size
		self.schedule = np.zeros((max_length, self.tss), dtype="float32")
		self.schedule_scale = self.tss
		basis = 10000
		for tta in range(max_length):
			for i in range(self.tss // 2):
				self.schedule[tta][2 * i] = mt.sin(1.0 * tta / mt.pow(basis, 2.0 * i / self.tss))
				self.schedule[tta][2 * i + 1] = mt.cos(1.0 * tta / mt.pow(basis, 2.0 * i / self.tss))

	def __getitem__(self, index):
		data = self.data[index]

		gating_index = np.random.randint(len(data)-self.gating_length)

		gating_seq = data[gating_index:gating_index+self.gating_length]

		# pre_character, future character, pre camera
		r_character = np.array(data[:, :9])
		r_camera = np.array(data[:, -5:])

		index = 1+np.random.randint(len(data)-self.max_length-1)
		character = r_character[index:index + self.max_length]
		camera = r_camera[index - 1:index + self.max_length]

		Schedule = np.zeros((self.max_length, self.schedule_scale), dtype="float32")
		Target = np.zeros((self.max_length, 14), dtype="float32")
		Mask = np.zeros((self.max_length, 1), dtype="float32")

		segment = np.random.randint(10)+1

		step = self.max_length // segment

		now = 0
		for i in range(segment):
			if i == segment-1:
				step = self.max_length-now
			for j in range(step):
				Target[now+j] = data[index+now+step-1]
			Schedule[now:now+step] = np.flip(self.schedule[:step], 0)
			now += step
			Mask[now - 1] = 1

		if np.random.rand() < 0.5:
			Target[0] = data[index]
			Schedule[0] = self.schedule[0]
			Mask[0] = 1

		return gating_seq, character, camera, Mask, Schedule, Target, r_camera[index:index+5].reshape(-1)

	def __len__(self):
		return len(self.data)
