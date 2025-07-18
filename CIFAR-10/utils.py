## Create the dataset and add transforms
## Create gradient_penalty function. Add
## function to plot the generated images
## and compare against original datasets

import torch.utils.data.dataloader
from config import *
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

DATASET = CIFAR10(root='CIFAR-10/dataset',train=True,
                  download=True,transform=transform)

def gradient_penalty(critic, real, fake, label, device):
    Batch,C,H,W = real.shape
    alpha = torch.rand(Batch,1,1,1).repeat(1,C,H,W).to(device)
    '''
    Idea to apply critic on interpolated
    image (combining both real and fake)
    and force the norm to be as close to
    1 as possible This ensures Lipschitz
    constraint.
    '''
    interpolated_images = (alpha*real) + ((1-alpha)*fake)
    mixed_scores = critic(interpolated_images,label)
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm-1)**2)

def plot(real, fake):
    real_np = make_grid(real*0.5 + 0.5).permute(1,2,0)
    fake_np = make_grid(fake*0.5 + 0.5).permute(1,2,0)
    plt.imshow(fake_np.cpu().numpy())
    plt.title('Fake')
    plt.show()
    plt.imshow(real_np.cpu().numpy())
    plt.title('Real')
    plt.show()

def eval(real, noise, label, critic, generator, epoch):
    fake =  generator(noise,label)
    fake_loss = critic(fake,label)
    real_loss = critic(real,label)
    '''
    Critic loss are the objective target
    that the model wants to minimize and
    to it we also add a gradient penalty
    term. Generator loss is the negative
    fake_loss.
    '''
    gp = gradient_penalty(critic,real,fake,label,DEVICE)
    critic_loss = -(real_loss.mean() - fake_loss.mean()) + LAMBDA_GP*gp
    print(f'Epoch:{epoch}/{EPOCHS} |\t Critic Loss: {critic_loss:.4f}')
    plot(real, fake)

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Linear,nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d,nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data,mean=0.0,std=0.02)