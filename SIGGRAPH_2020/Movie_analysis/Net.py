import torch
import math as mt
from torch import nn

class combined_CNN(nn.Module):
    def __init__(self, seq_length, channels):
        super(combined_CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            )

        self.mid_dim = int(seq_length / 4) * 128

        self.fc_toric_ABY = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            )

        self.fc_toric_theta = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )

        self.fc_toric_phi = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )

        self.fc_dist = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )

        self.fc_orientation = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            )

        self.fc_shoulder = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            )

    def forward(self, skelet):
        batch_size = len(skelet[0])
        
        # Mid0 = self.conv1(skelet[0]).view(batch_size, self.mid_dim)
        Mid1 = self.conv2(skelet[1]).view(batch_size, self.mid_dim)
        Mid2 = self.conv3(skelet[2]).view(batch_size, self.mid_dim)
        Mid3 = self.conv4(skelet[3]).view(batch_size, self.mid_dim)

        # output_toric_ABY    = self.fc_toric_ABY(Mid0)
        output_toric_theta  = self.fc_toric_theta(Mid1)
        output_toric_phi    = self.fc_toric_phi(Mid2)
        output_dist         = self.fc_dist(Mid3)
        output_orientation  = self.fc_orientation(Mid3)
        output_shoulder     = self.fc_shoulder(Mid3)

        output = torch.cat((output_toric_theta, output_toric_phi, output_dist,
            output_orientation, output_shoulder), 1)

        output = output.view(batch_size, -1)

        return output

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
                 input_size=1380,
                 output_size=150,
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

    def forward(self, global_seq):
        return self.Gate(global_seq)
