## Train the WGAN model on the MNIST dataset
## For 1 generator step, critic is trained 5
## times. Adam optimizer is used, with betas
## from WGAN-GP paper Weight are initialized
## to 0 mean and 0.02 standard deviation.

from utils import *
from config import *
from models import Critic, Generator

import torch
from torch.optim import Adam
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

"""
Create loader using Dataloader() Declare the
models and their optimizers. Hyperparameters
are referred using trial/error, and articles
Better ways for fine-tuning (◡ ‿ ◡ .)
"""
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

critic = Critic(IN_CHANNELS,CRITIC_CHANNEL_LIST,LABEL,CRITIC_EMBEDDING,IMAGE_SIZE).to(DEVICE)
generator = Generator(Z_DIM,GENERATOR_CHANNEL_LIST,LABEL,GENERATOR_EMBEDDING).to(DEVICE)
optimizer_critic = Adam(critic.parameters(),lr=LEARNING_RATE, betas=(0.0,0.9))
optimizer_gen = Adam(generator.parameters(),lr=LEARNING_RATE, betas=(0.0,0.9))

weight_initialization(critic)
weight_initialization(generator)

tls = []
for epoch in range(EPOCHS): ##[CONFIG]
    for i, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        ## train critic
        critic.train()
        generator.train()
        for _ in range(CRITIC_ITER):
            z = torch.randn(y.shape[0],Z_DIM).to(DEVICE)
            fake = generator(z,y)
            real_loss = critic(x,y)
            fake_loss = critic(fake,y)
            gp = gradient_penalty(critic,x,fake,y,DEVICE)
            critic_loss = -(real_loss.mean()-fake_loss.mean()) + LAMBDA_GP*gp
            optimizer_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_critic.step()
        
        fake_loss = -torch.mean(critic(fake,y))
        optimizer_gen.zero_grad()
        fake_loss.backward()
        optimizer_gen.step()
    
    critic.eval()
    generator.eval()
    x,y = next(iter(train_loader))
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    z = torch.randn(y.shape[0],Z_DIM).to(DEVICE)
    tls.append(eval(x,y,z,critic,generator))