## Training part: Based on config params
## train the given models and plot these
## progression in the generated and real
## images. Use WGAN-GP setup

from utils import *
from config import *
from models import Critic, Generator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

'''
Declare models and initialize weight to
be normal distributed with 0 mean, 0.02
standard deviation.
'''

critic = Critic(CHN,LABEL,EMBEDDING,IMSIZE).to(DEVICE)
gen = Generator(NOISE,GEN_CHN,LABEL,GEN_EMBEDDING).to(DEVICE)
optim_gen = Adam(gen.parameters(), LEARNING_RATE, betas = (0.5,0.999))
optim_critic = Adam(critic.parameters(), LEARNING_RATE, betas = (0.5,0.999))

weight_initialization(critic)
weight_initialization(gen)

dataloader = DataLoader(DATASET,BATCH_SIZE,shuffle=True)
for epoch in range(EPOCHS):
    for _, (x,y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        ## training
        gen.train()
        critic.train()
        for _ in range(CRITIC_ITER):
            z = torch.randn(x.shape[0],NOISE).to(DEVICE)
            fake = gen(z,y)
            real_loss = critic(x,y)
            fake_loss = critic(fake,y)
            gp = gradient_penalty(critic,x,fake,y,DEVICE)
            critic_loss = -(real_loss.mean()-fake_loss.mean())+LAMBDA_GP*gp
            optim_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optim_critic.step()
        gen_loss = -critic(fake,y).mean()
        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()
    
    ## evaluation
    gen.eval()
    critic.eval()
    x,y = next(iter(dataloader))
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    z = torch.randn(x.shape[0],NOISE,device=DEVICE)
    eval(x,z,y,critic,gen,epoch=epoch)