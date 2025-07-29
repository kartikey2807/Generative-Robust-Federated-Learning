## This code work with MNIST dataset
## This is un-conditional generation
## Have to condition these on labels
import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

BATCH = 64
SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## IMPD
LEARNING_RATE = 0.0001
EPOCHS = 5
TIMESTEPS = 1000

## Load the MNIST dataset and define
## the U-net model for denoising The
## models is supposed to predict the
## noise added to the image. Add the
## noise scheduler (betas).

transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])])
train_set = MNIST(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_set,batch_size=BATCH,shuffle=True)

Unet = UNet2DModel(
    sample_size=SIZE,
    in_channels=1,
    out_channels=1,
    time_embedding_type="positional",
    block_out_channels=(128,128,256,512),
    layers_per_block=2).to(DEVICE)

scheduler = DDPMScheduler(
    num_train_timesteps=TIMESTEPS,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon")

optimizer = torch.optim.Adam(Unet.parameters(),lr=LEARNING_RATE)
loss_term = nn.MSELoss()

## During training sample a batch of
## images, noise terms, and timestep
## Pass the noisy image to U-net and
## compute loss between true noise &
## predicted noise.Scheduler help in
## converting 'pure' image to noisy.

for epoch in range(EPOCHS):
    Unet.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, _ in pbar:
        ## sample images, noise, and
        ## timesteps. Pass the noisy
        ## images to U-net & compute
        ## MSE loss.
        images = images.to(DEVICE)
        noise  = torch.randn_like(images).to(DEVICE)
        timesteps = torch.randint(0,TIMESTEPS,(images.shape[0],)).to(DEVICE)
        timesteps = timesteps.long()
        noiseImgs = scheduler.add_noise(images,noise,timesteps)
        noisePred = Unet(noiseImgs,timesteps).sample
        loss = loss_term(noisePred,noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

pipeline = DDPMPipeline(unet=Unet, scheduler=scheduler)
pipeline.to(DEVICE)
samples  = pipeline(batch_size=BATCH, generator=torch.manual_seed(0)).images

fig, axes = plt.subplots(8,8,figsize=(15,15))
for i in range(8):
    for j in range(8):
        axes[i,j].imshow(samples[i*8+j], cmap='gray')  ## Cannot show labels
        axes[i,j].axis('off')
plt.tight_layout()
plt.show()