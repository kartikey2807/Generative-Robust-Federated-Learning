## Training WGAN model but using spectral normalization
## on the critic instead of weight clipping (to satisfy
## 1-Lipschitz constraint). Critic is trained 5 time to
## 1 generator update. Condition the GAN on labels Plot
## the critic loss with respect to epochs. Save generator
## model.

from models import Generator, Critic
from config import *
from utils import ToyDataset, eval

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import warnings
warnings.filterwarnings("ignore")
dataset = ToyDataset(n_samples=1000, noise=0.15, factor=0.3)
loader  = DataLoader(dataset, BATCH_SIZE,  shuffle=True)
gen = Generator(Z_DIM,GENDIM,EMBEDDING,LABEL).to(DEVICE)
critic = Critic(INPUT,HIDDEN,EMBEDDING,LABEL).to(DEVICE)
optimizer_gen = Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,0.9)) ## taken from WGAN paper
optimizer_critic = Adam(critic.parameters(),lr=LEARNING_RATE,betas=(0.5,0.9))

tloss = []
from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)): # mentioned in config
    for i, (x,y) in enumerate(loader):
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y, dtype=torch.long).to(DEVICE)

        gen.train()
        critic.train()
        # train critic
        for _ in range(CRITIC_ITERS):
            z = torch.randn(y.shape[0],Z_DIM).to(DEVICE)
            fake = gen(z,y)
            real_loss = critic(x,y)
            fake_loss = critic(fake,y)
            loss_for_critic = -(torch.mean(real_loss)-torch.mean(fake_loss))
            optimizer_critic.zero_grad()
            loss_for_critic.backward(retain_graph =True)
            optimizer_critic.step()
        
        # train generator
        sample_dist = torch.mean(torch.cdist(fake,fake,p=2)) ## [divergence]
        loss_for_generator = -(torch.mean(critic(fake,y)) + IMP*sample_dist)
        optimizer_gen.zero_grad()
        loss_for_generator.backward()
        optimizer_gen.step()
    
    if (epoch+1)%100 != 0:
        continue
    # evaluate on the models; given noise and real value
    gen.eval()
    critic.eval()
    p = torch.tensor(dataset.X[:500], dtype=torch.float32).to(DEVICE)
    q = dataset.y[:500].long().to(DEVICE)
    testnoise = torch.randn(q.shape[0],Z_DIM).to(DEVICE)
    tloss.append(eval(gen,critic,p,testnoise,q,epoch+1))

## Plot the loss with respect to epochs. Expect the loss
## to decrease and slowly converge to zero.
import matplotlib.pyplot as plt
plt.plot(range(100, EPOCHS+1, 100), tloss)
plt.xlabel('Epochs')
plt.ylabel('Critic Loss')
plt.title('Critic Loss vs Epochs')
plt.show()
########################################################