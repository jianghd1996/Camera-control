import torch
import torch.nn.parallel
from torch import nn
import math as mt

class Gating(nn.Module):
    def __init__(self,
        num_experts,
        input_size = 14,
        hidden_size = 512,
        num_layers = 1):

        super(Gating, self).__init__()

        self.num_experts = num_experts
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.RNN = torch.nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        self.fc = nn.Sequential(    
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=1),
            )

    def initHidden(self, batch_size):
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        return (h0, c0)

    def forward(self, input):
        hidden = self.initHidden(256)
        self.RNN.flatten_parameters()

        output, hidden = self.RNN(input, hidden)

        output = output.transpose(0, 1)[-1]

        return self.fc(output)

class linear(nn.Module):
    '''
    Modified linear layer in pytorch
    Each linear layer have num_experts weights
    Blend the weight with coefficients first and then multiply the matrix
    '''
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
        bias   = torch.matmul(controlweights, self.bias.view(self.num_experts, -1))

        weight = weight.view(-1, self.out_features, self.in_features)
        bias   = bias.view(-1, self.out_features)

        m = torch.matmul(weight, input.unsqueeze(2)).squeeze(2)

        return m + bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Prediction(nn.Module):
    def __init__(self, num_experts, input_size,output_size, hidden_size):
        super(Prediction, self).__init__()

        self.num_experts = num_experts
        self.fc1 = nn.Sequential(
            linear(input_size, hidden_size, num_experts),
            nn.ELU(),
            )
        self.fc2 = nn.Sequential(
            linear(hidden_size, hidden_size, num_experts),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            linear(hidden_size, output_size, num_experts),
            )
        
        self.Gate = Gating( num_experts = num_experts)

    def forward(self, global_seq, local_seq):
        self.controlweights = self.Gate(global_seq)

        Mid = self.fc1([local_seq, self.controlweights])
        Mid = self.fc2([Mid, self.controlweights])
        return self.fc3([Mid, self.controlweights])