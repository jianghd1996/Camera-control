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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class Transformer(nn.Module):
    def __init__(self, n_feature, n_label, latent_dim=256,
                 num_heads=4, ff_size=1024, dropout=0.1, activation='gelu',
                 num_layers=4, sliding_wind=300):
        super(Transformer, self).__init__()

        self.n_feature = n_feature
        self.n_label = n_label
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.input_process = nn.Linear(self.n_feature, self.latent_dim)

        seqTransEncoderlayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead = self.num_heads,
                                                          dim_feedforward = self.ff_size,
                                                          dropout = self.dropout,
                                                          activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderlayer,
                                                     num_layers = self.num_layers)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.output_process = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.ReLU()
        )
        self.pred = nn.Sequential(
            nn.Linear(sliding_wind, n_label),
            # nn.Softmax(dim=1),
        )


    def forward(self, x):
        bs = len(x)
        x = self.input_process(x.permute(1, 0, 2))

        xseq = self.sequence_pos_encoder(x)
        xseq = self.seqTransEncoder(xseq)
        xseq = self.output_process(xseq).permute(1, 0, 2)

        xseq = xseq.view(bs, -1)

        return self.pred(xseq)

    def forward_feature(self, x):
        bs = len(x)
        x = self.input_process(x.permute(1, 0, 2))

        xseq = self.sequence_pos_encoder(x)
        xseq = self.seqTransEncoder(xseq)
        xseq = self.output_process(xseq).permute(1, 0, 2)

        return xseq.view(bs, -1)

import torch.utils.data as data
class camdataset(data.Dataset):
    def __init__(self, cam, label):
        self.cam = cam
        self.label = label

    def __getitem__(self, index):
        d = self.cam[index]
        data = np.concatenate((d, d[-1:].repeat(300-len(d), 0)), 0)
        return np.array(data, dtype="float32"), self.label[index]

    def __len__(self):
        return len(self.cam)


def train_mnist():
    data = np.load("data.npy", allow_pickle=True)[()]

    d = np.concatenate(data["train_cam"]+data["test_cam"], 0)
    Mean, Std = np.mean(d, 0), np.std(d, 0)

    np.save("Mean_Std", {"Mean": Mean, "Std": Std})

    for i in range(len(data["train_cam"])):
        data["train_cam"][i] = (data["train_cam"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    for i in range(len(data["test_cam"])):
        data["test_cam"][i] = (data["test_cam"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    # hardcoding these here
    n_epoch = 1000
    batch_size = 128
    device = "cuda:0"
    n_feature = 5
    n_label = 6
    lrate = 1e-4
    save_model = True
    save_dir = './result/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    criterion = torch.nn.CrossEntropyLoss()
    trans = Transformer(n_feature=n_feature, n_label=n_label)
    trans.to(device)

    optim = torch.optim.Adam(trans.parameters(), lr=lrate)

    dataloader = DataLoader(camdataset(data['train_cam'], data['train_label']), batch_size=batch_size, shuffle=True, num_workers=5)
    testloader = DataLoader(camdataset(data['test_cam'], data['test_label']), batch_size=batch_size, shuffle=False, num_workers=5)

    if not os.path.exists("result"):
        os.mkdir("result")

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)

        trans.train()
        correct = 0
        total = 0
        for cam, label in pbar:
            cam = cam.to(device)
            label = label.to(device)

            pred_v = trans(cam)

            predictions = torch.argmax(pred_v, dim=1)
            correct += torch.sum(predictions == label).item()
            total += len(predictions)

            optim.zero_grad()
            loss = criterion(pred_v, label)
            loss.backward()

            pbar.set_description(f"training acc: {100.0 * correct/total:.4f}")
            optim.step()

        trans.eval()
        correct = 0
        total = 0
        for cam, label in testloader:
            cam = cam.to(device)
            label = label.to(device)

            pred_v = trans(cam)
            predictions = torch.argmax(pred_v, dim=1)

            correct += torch.sum(predictions == label)
            total += len(predictions)
        print("evaluation accuracy : {}".format(1.0 * correct / total))

        torch.save(trans.state_dict(), save_dir + f"latest.pth")
        if save_model and ep % 100 == 0:
            torch.save(trans.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def eval_mnist(file_name):
    if not os.path.exists("Mean_Std.npy"):
        data = np.load("data.npy", allow_pickle=True)[()]

        d = np.concatenate(data["train_cam"] + data["test_cam"], 0)
        Mean, Std = np.mean(d, 0), np.std(d, 0)
        np.save("Mean_Std", {"Mean":Mean, "Std":Std})
    
    d = np.load("Mean_Std.npy", allow_pickle=True)[()]
    Mean, Std = d["Mean"], d["Std"]

    data = np.load(file_name+".npy", allow_pickle=True)[()]

    for i in range(len(data["result"])):
        data["result"][i] = (data["result"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    device = "cuda:0"
    n_feature = 5
    n_label = 6

    trans = Transformer(n_feature=n_feature, n_label=n_label)
    trans.to(device)

    # optionally load a model
    trans.load_state_dict(torch.load("./result/latest.pth"))

    testloader = DataLoader(camdataset(data['result'], data['label']), batch_size=8, num_workers=5)

    correct = 0
    total = 0
    t = [0] * 10
    f = [0] * 10
    trans.eval()
    with torch.no_grad():
        for cam, label in tqdm(testloader):
            cam = cam.to(device)
            label = label.to(device)

            pred_v = trans(cam)
            predictions = torch.argmax(pred_v, dim=1)

            correct += torch.sum(predictions == label)
            total += len(predictions)

            for i in range(len(predictions)):
                if predictions[i] == label[i]:
                    t[label[i]] += 1
                else:
                    f[label[i]] += 1

    print("gen accuracy : {}/{}={} ".format(correct, total, 1.0 * correct / total))
    for i in range(n_label):
        print("{} {} {}".format(i, t[i], t[i]+f[i]))

def process_feature(file_list):
    data = np.load("data.npy", allow_pickle=True)[()]

    d = np.concatenate(data["train_cam"] + data["test_cam"], 0)
    Mean, Std = np.mean(d, 0), np.std(d, 0)

    for i in range(len(data["train_cam"])):
        data["train_cam"][i] = (data["train_cam"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    for i in range(len(data["test_cam"])):
        data["test_cam"][i] = (data["test_cam"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    device = "cuda:0"
    n_feature = 5
    n_label = 6

    trans = Transformer(n_feature=n_feature, n_label=n_label)
    trans.to(device)

    # optionally load a model
    trans.load_state_dict(torch.load("./result/latest.pth"))

    trans.eval()

    d = dict()

    testloader = DataLoader(camdataset(data['train_cam'], data['train_label']), batch_size=8, num_workers=5)

    feature = []

    with torch.no_grad():
        for cam, label in tqdm(testloader):
            cam = cam.to(device)

            pred_v = trans.forward_feature(cam).detach().cpu().numpy()

            for v in pred_v:
                feature.append(v)

        d["train_data"] = feature

    testloader = DataLoader(camdataset(data['test_cam'], data['test_label']), batch_size=8, num_workers=5)

    feature = []

    with torch.no_grad():
        for cam, label in tqdm(testloader):
            cam = cam.to(device)

            pred_v = trans.forward_feature(cam).detach().cpu().numpy()

            for v in pred_v:
                feature.append(v)

        d["test_data"] = feature


    for file in file_list:
        data = np.load(file+".npy", allow_pickle=True)[()]

        for i in range(len(data["result"])):
            data["result"][i] = (data["result"][i] - Mean[None, :]) / (Std[None, :] + 1e-8)

        testloader = DataLoader(camdataset(data['result'], data['label']), batch_size=8, num_workers=5)

        feature = []

        with torch.no_grad():
            for cam, label in tqdm(testloader):
                cam = cam.to(device)

                pred_v = trans.forward_feature(cam).detach().cpu().numpy()

                for v in pred_v:
                    feature.append(v)

            d[file] = feature

    np.save("feature", d)


if __name__ == "__main__":
    train_mnist()
    #
    # eval_mnist()

    # process_feature()
