import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
import os
from tqdm import tqdm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(data):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma

def calculate_diversity(data, first_indices, second_indices):
    diversity = 0

    d = torch.FloatTensor(data)

    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(d[first_idx, :], d[second_idx, :])

    diversity /= len(first_indices)
    return diversity

d = np.load("feature.npy", allow_pickle=True)[()]

d0 = d["train_data"]
d1 = d["test_data"]
d2 = d["gen_T5"]
d3 = d["gen_GRU_T5"]
d4 = d["LSTM_Des"]
d5 = d["gen"]

Mean, Std = np.mean(d0, 0), np.std(d0, 0)
d0 = [(v - Mean[None, :]) / Std[None, :] for v in d0]
d1 = [(v - Mean[None, :]) / Std[None, :] for v in d1]
d2 = [(v - Mean[None, :]) / Std[None, :] for v in d2]
d3 = [(v - Mean[None, :]) / Std[None, :] for v in d3]
d4 = [(v - Mean[None, :]) / Std[None, :] for v in d4]
d5 = [(v - Mean[None, :]) / Std[None, :] for v in d5]

if not os.path.exists("viz"):
    os.mkdir("viz")


d0 = np.array([v.flatten() for v in d0])
d1 = np.array([v.flatten() for v in d1])
d2 = np.array([v.flatten() for v in d2])
d3 = np.array([v.flatten() for v in d3])
d4 = np.array([v.flatten() for v in d4])
d5 = np.array([v.flatten() for v in d5])

print("Diversity")

diversity_times = 10000
num_motions = len(d1)
first_indices = np.random.randint(0, num_motions, diversity_times)
second_indices = np.random.randint(0, num_motions, diversity_times)

print(calculate_diversity(d1, first_indices, second_indices))
print(calculate_diversity(d2, first_indices, second_indices))
print(calculate_diversity(d3, first_indices, second_indices))
print(calculate_diversity(d4, first_indices, second_indices))
print(calculate_diversity(d5, first_indices, second_indices))

print("Diversity with action label")

d = np.load("data.npy", allow_pickle=True)[()]

label = dict()
for i in range(6):
    label[i] = []
for i in range(len(d['test_label'])):
    label[d['test_label'][i]].append(i)

diversity_times = 1000
first_indices = []
second_indices = []
for i in range(6):
    idx = np.random.randint(0, len(label[i]), diversity_times)
    for j in idx:
        first_indices.append(label[i][j])
    idx = np.random.randint(0, len(label[i]), diversity_times)
    for j in idx:
        second_indices.append(label[i][j])

import random
print(random.shuffle(second_indices))

print(calculate_diversity(d1, first_indices, second_indices))
print(calculate_diversity(d2, first_indices, second_indices))
print(calculate_diversity(d3, first_indices, second_indices))
print(calculate_diversity(d4, first_indices, second_indices))
print(calculate_diversity(d5, first_indices, second_indices))


print("FID with training")

mu0, sigma0 = calculate_activation_statistics(d0)
mu1, sigma1 = calculate_activation_statistics(d1)
mu2, sigma2 = calculate_activation_statistics(d2)
mu3, sigma3 = calculate_activation_statistics(d3)
mu4, sigma4 = calculate_activation_statistics(d4)
mu5, sigma5 = calculate_activation_statistics(d5)

print(calculate_frechet_distance(mu0, sigma0, mu1, sigma1))
print(calculate_frechet_distance(mu0, sigma0, mu2, sigma2))
print(calculate_frechet_distance(mu0, sigma0, mu3, sigma3))
print(calculate_frechet_distance(mu0, sigma0, mu4, sigma4))
print(calculate_frechet_distance(mu0, sigma0, mu5, sigma5))