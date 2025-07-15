## Train WGAN-GP model with adam optim
## Run for about 20000 epochs and plot
## loss wrt. epochs Note the generated
## distribution.

from utils import *
from config import *
from models import Generator, Critic

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import warnings
warnings.filterwarnings("ignore")
dataset = ToyDataset(n_samples=1000,noise=0.15,factor=0.3)
loader  = DataLoader(dataset, BATCH_SIZE,  shuffle=True)
gen = Generator(Z_DIM,GENDIM,EMBEDDING,LABEL).to(DEVICE)
critic = Critic(INPUT,HIDDEN,EMBEDDING,LABEL).to(DEVICE)
optimizer_gen = Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,0.9)) ## taken from WGAN paper
optimizer_critic = Adam(critic.parameters(),lr=LEARNING_RATE,betas=(0.5,0.9))

tloss = []
from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    for i, (x,y) in enumerate(loader):
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y, dtype=torch.long).to(DEVICE)
        
        # train critic
        gen.train()
        critic.train()
        for _ in range(CRITIC_ITERS):
            z = torch.randn(y.shape[0],Z_DIM).to(DEVICE)
            fake = gen(z,y)
            real_loss = critic(x,y)
            fake_loss = critic(fake,y)
            gp = gradient_penalty(critic,fake,x,y,device=DEVICE)
            loss_for_critic = -(real_loss.mean()-fake_loss.mean()) + LAMBDA_GP*gp
            optimizer_critic.zero_grad()
            loss_for_critic.backward(retain_graph =True)
            optimizer_critic.step()
        
        # train generator
        inter_sample_distance = torch.mean(torch.cdist(fake,fake,p=2))
        loss_for_generator = -(torch.mean(critic(fake,y)) + IMP*inter_sample_distance)
        ## objective now: is also to 
        ## increase intersample dist
        ## -ance, and so we decrease
        ## the negatives of the same
        optimizer_gen.zero_grad()
        loss_for_generator.backward()
        optimizer_gen.step()
    
    if (epoch+1)%100 != 0:
        continue

    gen.eval()
    critic.eval()
    p = torch.tensor(dataset.X[:500], dtype=torch.float32).to(DEVICE)
    q = dataset.y[:500].long().to(DEVICE)
    testnoise = torch.randn(q.shape[0],Z_DIM).to(DEVICE)
    tloss.append(eval(gen,critic,p,testnoise,q,epoch+1))

## Plot the loss with respect to epochs
## Expect the loss to decrease & slowly
## converge to zero.
import matplotlib.pyplot as plt
plt.plot(range(100, EPOCHS+1, 100), tloss)
plt.xlabel('Epochs')
plt.ylabel('Critic Loss')
plt.title('Critic Loss vs Epochs')
plt.show()