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
import random
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
    def __init__(self, n_feature, n_textemb, latent_dim=256,
                 num_heads=4, ff_size=1024, dropout=0.1, activation='gelu',
                 num_layers=4, cond_mask_prob=0.1):
        super(Transformer, self).__init__()

        self.n_feature = n_feature
        self.n_textemb = n_textemb
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob

        self.embed_text = nn.Linear(self.n_textemb, self.latent_dim)

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

        self.output_process = nn.Linear(self.latent_dim, self.n_feature)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, emb_text, timesteps, force_mask=False):
        emb_time = self.embed_timestep(timesteps)

        emb_text = self.embed_text(self.mask_cond(emb_text, force_mask=force_mask))
        emb = (emb_time + emb_text)

        x = self.input_process(x.permute(1, 0, 2))

        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[1:]

        return self.output_process(output).permute(1, 0, 2)



def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

        self.count = [0] * n_T

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        for t in _ts:
            self.count[t] += 1

        x_t = (
            self.sqrtab[_ts, None, None] * x
            + self.sqrtmab[_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts))

    def sample(self, n_sample, c, size, device, guide_w):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        if c.shape[0] == 1:
            c_i = c.repeat(n_sample, 1).float()
        else:
            c_i = c.float()

        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.tensor(i).to(device).repeat(n_sample)

            # split predictions and compute weighting
            eps1 = self.nn_model(x_i, c_i, t_is)
            eps2 = self.nn_model(x_i, c_i, t_is, force_mask=True)
            eps = eps2 + guide_w * (eps1 - eps2)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0


            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

        return x_i



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
        data["cam"][i] = (data["cam"][i] - Mean[None, :]) / (Std[None, :]+1e-8)

    # hardcoding these here
    n_epoch = 20000
    batch_size = 256
    n_T = 1000 # 500
    device = "cuda:0"
    n_feature = 5
    n_textemb = 512
    lrate = 1e-4
    save_model = True
    save_dir = './weight/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ddpm = DDPM(nn_model=Transformer(n_feature=n_feature, n_textemb=n_textemb), betas=(1e-4, 0.02), n_T=n_T, device=device)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    dataloader = DataLoader(camdataset(data['cam'], data['info']), batch_size=batch_size, shuffle=True, num_workers=5)

    if not os.path.exists("result"):
        os.mkdir("result")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            with torch.no_grad():
                c = clip.tokenize(c, truncate=True).to(device)
                c = model.encode_text(c).detach()

            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        torch.save(ddpm.state_dict(), save_dir + f"latest.pth")
        if save_model and ep % 100 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


def gen():
    if not os.path.exists("Mean_Std.npy"):
        data = np.load("data.npy", allow_pickle=True)[()]

        d = np.concatenate(data["cam"], 0)
        Mean, Std = np.mean(d, 0), np.std(d, 0)
        np.save("Mean_Std", {"Mean": Mean, "Std": Std})

    d = np.load("Mean_Std.npy", allow_pickle=True)[()]
    Mean, Std = d["Mean"], d["Std"]

    n_T = 1000  # 500
    device = "cuda:0"
    n_feature = 5
    n_textemb = 512

    ddpm = DDPM(nn_model=Transformer(n_feature=n_feature, n_textemb=n_textemb), betas=(1e-4, 0.02), n_T=n_T,
                device=device)
    ddpm.to(device)

    # optionally load a model
    ddpm.load_state_dict(torch.load("./weight/latest.pth"))

    if not os.path.exists("gen"):
        os.mkdir("gen")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = ["The camera pans to the character. The camera switches from right front view to right back view. The character is at the middle center of the screen. The camera shoots at close shot."]

    result = []

    def smooth(x, winds=10, T=4):
        if T == 0:
            return x
        n_x = np.array(x)
        for i in range(len(x)):
            n_x[i] = np.mean(x[max(0, i - winds):min(len(x), i + winds), :], 0)
        return smooth(n_x, T=T - 1)

    with torch.no_grad():
        c = clip.tokenize(text, truncate=True).to(device)
        c = model.encode_text(c)

        sample = ddpm.sample(10, c, (300, n_feature), device, guide_w=2.0)
        sample = sample.detach().cpu().numpy()

        for j in range(len(sample)):
            s = smooth(sample[j] * Std[None, :] + Mean[None, :])
            result.append(s)
            with open("gen/{}.txt".format(j), "w") as f:
                for i in range(len(s)):
                    txt = ""
                    for k in range(5):
                        txt += str(s[i][k]) + " "
                    f.write(txt+"\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1]

    if mode == 'train':
        train()
    elif mode == 'gen':
        gen()
    else:
        print('Error, instruction {} is not in {train, gen}')

