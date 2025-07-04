## Create toy dataset with 2D feature and binary
## labels. Use sklearn to generate data and wrap
## in torch.utils.data.Dataset for training. Use
## DataLoader for batching and shuffling. Eval()
## is used to measure the performance of the GAN
## on data points. It plots as well.

## fine-tuning sklearn.datasets.make_circle  for
## creating samples.

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from tqdm import tqdm
def eval(gen, critic, x, z, y, epoch):
    fake = gen(z,y)
    real_loss = critic(x,y)
    fake_loss = critic(fake,y)
    critic_loss = torch.mean(real_loss)-torch.mean(fake_loss)
    gen_loss = torch.mean(fake_loss)
    tqdm.write(f"Epoch: {epoch}\t| Critic Loss: {critic_loss.item():.7f}\t|\
                Generator Loss: {gen_loss.item():.7f}")
    
    ## Plot the generated sample and real sample
    ## in 2D space.
    if epoch%3000 == 0:
        f = fake.detach().cpu().numpy()
        plot_samples(f=f,x=x.numpy(),y=y.numpy())
    return critic_loss.item()

def scatter(x, y, c, marker):
    plt.scatter(x,y,c=c,alpha=0.7,marker=marker)

def plot_samples(f, x, y):
    scatter(x[y==0,0], x[y==0,1], 'orange', 'o')
    scatter(x[y==1,0], x[y==1,1], 'yellow', 'o')
    scatter(f[y==0,0], f[y==0,1], 'blue', 'x') ## [generated]
    scatter(f[y==1,0], f[y==1,1], 'violet', 'x')
    plt.title('Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Real samples:0', 'Real samples:1', 'Generated samples:0', 'Generated samples:1'])
    plt.show()

class ToyDataset(Dataset):
    def __init__(self, n_samples, noise, factor):
        super().__init__()
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

## VISUALIZATION: Plot sample data points verify
## the dataset is generated correctly.
## dataset = ToyDataset(1000, 0.15, 0.3)
## X = dataset.X.numpy()
## y = dataset.y.numpy()
## plt.scatter(X[y==0,0], X[y==0,1], c='orange')
## plt.scatter(X[y==1,0], X[y==1,1], c='yellow')
## plt.title('Data Distribution')
## plt.xlabel('Feature 1')
## plt.ylabel('Feature 2')
## plt.legend(['Class 0', 'Class 1'])
## plt.show()
################################################