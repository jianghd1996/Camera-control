''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import clip

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_size=512, n_layer=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, bidirectional=bidirectional)

        self.encoder = nn.Sequential(nn.Linear(embed_size, hidden_size))

        self.decoder = nn.Sequential(nn.Linear(hidden_size, output_size))

        self.embed = nn.Sequential(nn.Linear(embed_size, embed_size))


    def initHidden(self, batch_size=1):
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size, requires_grad=False).cuda()
        c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size, requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, input, embed):
        bs, length, n_feat = input.shape

        embed = self.embed(embed).unsqueeze(1).repeat(1, length, 1)

        hidden = self.initHidden(bs)
        output, hidden = self.lstm(embed, hidden)

        return self.decoder(output)

import torch.utils.data as data
class camdataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        text = np.random.choice(self.label[index], np.random.randint(1, len(self.label[index])+1), replace=False)

        d = self.data[index]
        d = np.concatenate((d, d[-1:].repeat(300-len(d), 0)), 0)

        return np.array(d, dtype="float32"), " ".join(text)

    def __len__(self):
        return len(self.data)


def train():
    data = np.load("data.npy", allow_pickle=True)[()]

    d = np.concatenate(data["cam"], 0)
    Mean, Std = np.mean(d, 0), np.std(d, 0)

    for i in range(len(data["cam"])):
        data["cam"][i] = (data["cam"][i] - Mean[None, :]) / (Std[None, :] + 1e-8)

    # hardcoding these here
    n_epoch = 1000
    batch_size = 128
    device = "cuda:0"
    n_feature = 5
    lrate = 1e-4
    save_model = True
    save_dir = './result/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    criterion = torch.nn.MSELoss()
    trans = LSTM(input_size=n_feature, hidden_size=512, output_size=n_feature)
    trans.to(device)

    optim = torch.optim.Adam(trans.parameters(), lr=lrate)

    dataloader = DataLoader(camdataset(data['cam'], data['info']), batch_size=batch_size, shuffle=True, num_workers=5)

    if not os.path.exists("result"):
        os.mkdir("result")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        trans.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            with torch.no_grad():
                c = clip.tokenize(c, truncate=True).to(device)
                c = model.encode_text(c).float().detach()

            loss = criterion(trans(x, c), x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        torch.save(trans.state_dict(), save_dir + f"latest.pth")
        if save_model and ep % 100 == 0:

            torch.save(trans.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def eval():
    if not os.path.exists("Mean_Std.npy"):
        data = np.load("data.npy", allow_pickle=True)[()]

        d = np.concatenate(data["cam"], 0)
        Mean, Std = np.mean(d, 0), np.std(d, 0)
        np.save("Mean_Std", {"Mean": Mean, "Std": Std})
    d = np.load("Mean_Std.npy", allow_pickle=True)[()]
    Mean, Std = d["Mean"], d["Std"]

    device = "cuda:0"
    n_feature = 5

    trans = LSTM(input_size=n_feature, hidden_size=512, output_size=n_feature)
    trans.to(device)

    # optionally load a model
    trans.load_state_dict(torch.load("./result/latest.pth"))

    if not os.path.exists("viz"):
        os.mkdir("viz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    d = np.load("test_prompt.npy", allow_pickle=True)[()]

    result = []
    for i in tqdm(range(0, len(d['info']), 100)):
        txt = d['info'][i:i + 100]
        text = [" ".join(v) for v in txt]

        with torch.no_grad():
            c = clip.tokenize(text, truncate=True).to(device)
            c = model.encode_text(c).float().detach()

            sample = trans(torch.zeros(len(c), 300, n_feature), c)
            sample = sample.detach().cpu().numpy()

            for j in range(len(text)):
                s = sample[j] * Std[None, :] + Mean[None, :]
                result.append(s)

        np.save("LSTM_test", {"result": result, "label": d["label"]})

if __name__ == "__main__":
    import sys
    mode = sys.argv[1]

    if mode == 'train':
        train()
    elif mode == 'eval':
        eval()
    else:
        print('Error, instruction {} is not in {train, eval}')
