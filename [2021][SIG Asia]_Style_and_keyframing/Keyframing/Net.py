import torch
import torch.nn.parallel
import torch.optim
from torch import nn
import math as mt
from collections import OrderedDict

class GateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gating_input_size, gating_hidden_size, gating_output_size,
                 n_layer=1, bidirectional=False):
        super(GateLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gating_input_size = gating_input_size
        self.gating_hidden_size = gating_hidden_size
        self.gating_output_size =gating_output_size
        self.n_layer = n_layer
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, bidirectional=bidirectional)

        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),)

        self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, output_size))

        self.Gating = torch.nn.LSTM(input_size=gating_input_size,
                                 hidden_size=gating_hidden_size,
                                 num_layers=n_layer,
                                 batch_first=True, bidirectional=bidirectional)

        gru_hidden_size = self.n_layer * self.num_directions * self.hidden_size
        self.hidden_encoder = nn.Sequential(nn.Linear(5*5+4, gru_hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(gru_hidden_size, gru_hidden_size))

        self.fc = nn.Sequential(
            nn.Linear(gating_hidden_size, gating_output_size),
            # nn.Softmax(dim=1),
        )

    def initHidden(self, start_traj):
        batch_size = len(start_traj)
        return self.hidden_encoder(start_traj).view(self.n_layer * self.num_directions, batch_size, self.hidden_size)

    def GatingHidden(self, batch_size=1):
        h0 = torch.zeros(self.n_layer, batch_size, self.gating_hidden_size, requires_grad=False).cuda()
        c0 = torch.zeros(self.n_layer, batch_size, self.gating_hidden_size, requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, input, hidden=None):
        if not torch.is_tensor(hidden):
            hidden = self.GatingHidden(len(input))
            self.Gating.flatten_parameters()
            output, hidden = self.Gating(input, hidden)
            output = output.transpose(0, 1)
            return self.fc(output)

        feature = self.encoder(input)
        output, hidden = self.lstm(feature.unsqueeze(1), hidden)

        return self.decoder(output.squeeze(1)), hidden
