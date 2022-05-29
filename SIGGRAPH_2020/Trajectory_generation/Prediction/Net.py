import cv2
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm
from torch import nn
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import math as mt
import sys

class Shared_bottom(nn.Module):
    def __init__(self, input_size, label_num, output_size):
        super(Shared_bottom, self).__init__()
        self.Shared_input = nn.Sequential(nn.Linear(input_size, 1024),
                                          nn.ELU())
        self.Split_output = nn.ModuleList()
        for i in range(label_num):
            self.Split_output.append(nn.Sequential(nn.Linear(1024, 512),
                                                   nn.ELU(),
                                                   nn.Linear(512, output_size)))

    def forward(self, input, category):
        batch = len(category)
        output = []
        Mid = self.Shared_input(input)
        for i in range(batch):
            output.append(self.Split_output[category[i]](Mid[i]))
        return torch.stack(output, dim=0)

class FC(nn.Module):
    def __init__(self, input_length, output_length, max_style_num):
        super(FC, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.residual = nn.ModuleList([])
        for i in range(max_style_num):
            self.residual.append(nn.Sequential(
                nn.Linear(512, 128),
                nn.LeakyReLU(negative_slope=0.2),
            ))

        self.fc3 = nn.Sequential(
            nn.Linear(128, output_length),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, input_data, style_num):
        Mid = self.fc1(input_data)

        # if (style_num == 0):
        #     return self.fc3(self.fc2(Mid))
        return self.fc3(self.fc2(Mid) + self.residual[style_num](Mid))


class Gating(nn.Module):
    def __init__(self,
                 num_experts,
                 input_size=14,
                 hidden_size=512,
                 num_layers=1):
        super(Gating, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.RNN = torch.nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=1),
        )

    def initHidden(self, batch_size):
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return (c0, h0)

    def forward(self, input):
        self.hidden = self.initHidden(input.shape[0])
        self.RNN.flatten_parameters()

        output, self.hidden = self.RNN(input, self.hidden)

        output = output[0]
        output = self.fc(output)

        return output

class linear(nn.Module):

    def __init__(self, in_features, out_features, num_experts, bias=True):
        super(linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = torch.nn.Parameter(torch.Tensor(num_experts, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_experts, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=mt.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / mt.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        controlweights = input[1]
        input = input[0]

        weight = torch.matmul(controlweights, self.weight.view(self.num_experts, -1))
        bias = torch.matmul(controlweights, self.bias.view(self.num_experts, -1))

        weight = weight.view(-1, self.out_features, self.in_features)
        bias = bias.view(-1, self.out_features)

        m = torch.matmul(weight, input.unsqueeze(2)).squeeze(2)

        return m + bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Prediction(nn.Module):
    def __init__(self,
                 num_experts,
                 input_size=565,
                 output_size=5,
                 hidden_size=512,
                 ):
        super(Prediction, self).__init__()

        self.num_experts = num_experts

        self.fc1 = nn.Sequential(
            linear(input_size, hidden_size, num_experts),
            nn.ELU(),
            # nn.Dropout(p=0.2),
        )

        self.fc2 = nn.Sequential(
            linear(hidden_size, hidden_size, num_experts),
            nn.ELU(),
            # nn.Dropout(p=0.2),
        )

        self.fc3 = nn.Sequential(
            linear(hidden_size, output_size, num_experts),
        )

        self.Gate = Gating(num_experts=num_experts)

    def freeze_grad(self, nets):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.require_grads = False

    def style_parameters(self):
        self.fc1.require_grads = False
        self.fc2.require_grads = False
        # self.fc3.require_grads = False

        normal_weight = []
        normal_bias = []

        for m in self.Gate.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_bias.append(ps[1])

        for m in self.fc3.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_bias.append(ps[1])

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': 'normal_weight'},
            {'params': normal_bias, 'lr_mult': 1, 'decay_mult': 1,
             'name': 'normal_bias'},
        ]

    def forward(self, global_seq, local_seq):
        if not isinstance(local_seq, torch.Tensor):
            return self.Gate(global_seq)

        Mid = self.fc1([local_seq, global_seq])
        Mid = self.fc2([Mid, global_seq])
        return self.fc3([Mid, global_seq])
