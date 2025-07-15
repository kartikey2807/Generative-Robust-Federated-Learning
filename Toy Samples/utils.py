## Create toy dataset with 2D feature and
## binary labels. Use sklearn to generate
## data, wrap in torch.utils.data.Dataset
## Eval(.) is used to measure performance
## Add gradient penalty method.

from config import *

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
    gp = gradient_penalty(critic,fake,x,y,device=DEVICE)
    critic_loss = -(real_loss.mean() - fake_loss.mean()) + LAMBDA_GP*gp
    gen_loss = -torch.mean(fake_loss)
    tqdm.write(f"Epoch: {epoch}\t| Critic Loss: {critic_loss.item():.7f}\t|\
                Generator Loss: {gen_loss.item():.7f}")
    
    ## Plot the generated sample and real sample
    ## in 2D space.
    if epoch%1000 == 0:
        f = fake.detach().cpu().numpy()
        plot_samples(f,x.numpy(),y.numpy(),epoch)
    return critic_loss.item()

def scatter(x, y, c, marker):
    plt.scatter(x,y,c=c,alpha=0.7,marker=marker)

def plot_samples(f, x, y, epoch):
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

def gradient_penalty(critic, fake, real, label, device='cpu'):
    BATCH, FEATURE = real.shape
    alpha = torch.rand(BATCH,1).repeat(1,FEATURE).to(device)

    '''
    Create interpolated images, combining
    real and fake image Compute of Critic
    on this interpolated image and try to
    push it close to 1.
    '''
    interpolated_image = (alpha * real) + ((1-alpha) * fake)
    mixed_loss = critic(interpolated_image, label)
    gradient = torch.autograd.grad(
        outputs=mixed_loss,
        inputs=interpolated_image,
        grad_outputs=torch.ones_like(mixed_loss),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm-1)**2)