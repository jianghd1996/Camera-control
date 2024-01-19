import torch
from torch import nn


class combined_CNN(nn.Module):
    def __init__(self, config):
        super(combined_CNN, self).__init__()

        # be used to predict pA, pB, pY, which can directly get from 2D character head position
        # self.conv1 = self.conv(config)

        self.conv2 = self.conv(config)
        self.conv3 = self.conv(config)
        self.conv4 = self.conv(config)

        self.mid_dim = int(config.seq_length / 4) * 128

        self.fc_toric_theta = self.fc(config, self.mid_dim, 1)

        self.fc_toric_phi = self.fc(config, self.mid_dim, 1)

        self.fc_dist = self.fc(config, self.mid_dim, 1)

        self.fc_orientation = self.fc(config, self.mid_dim, 2)

        self.fc_shoulder = self.fc(config, self.mid_dim, 4)

    def conv(self, config):
        return nn.Sequential(
            nn.Conv1d(in_channels=config.channels[0], out_channels=config.channels[1], kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(config.channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=config.channels[1], out_channels=config.channels[2], kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(config.channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

    def fc(self, config, mid_dim, out_dim):
        return nn.Sequential(
            nn.Linear(mid_dim, config.fc_dim),
            nn.ReLU(),
            nn.Linear(config.fc_dim, out_dim),
            )

    def forward(self, skelet):
        batch_size = len(skelet[0])
        Mid1 = self.conv2(skelet[1]).view(batch_size, self.mid_dim)
        Mid2 = self.conv3(skelet[2]).view(batch_size, self.mid_dim)
        Mid3 = self.conv4(skelet[3]).view(batch_size, self.mid_dim)
        output_toric_theta  = self.fc_toric_theta(Mid1)
        output_toric_phi    = self.fc_toric_phi(Mid2)
        output_dist         = self.fc_dist(Mid3)
        output_orientation  = self.fc_orientation(Mid3)
        output_shoulder     = self.fc_shoulder(Mid3)

        output = torch.cat((output_toric_theta, output_toric_phi, output_dist,
            output_orientation, output_shoulder), 1)

        output = output.view(batch_size, -1)

        return output
