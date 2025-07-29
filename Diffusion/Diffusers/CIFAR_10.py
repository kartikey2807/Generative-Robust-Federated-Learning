## Code uncondtional DDPMs gen model
## for CIFAR-10 dataset. Again, need
## need to condition these on labels

import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

## Declare hyper-parameters
BATCH = 64
SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## DEVICE
LEARNING_RATE = 0.0001
EPOCHS = 75
TIMESTEPS = 1000

## Load the CIFAR-10 dataset and set
## the noise schedulers (beta terms)
## Declare the U-net model that will
## predict the noise added to images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
train_set = CIFAR10(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_set,batch_size=BATCH,shuffle=True)

Unet = UNet2DModel(
    sample_size=SIZE,
    in_channels=3,
    out_channels=3,
    time_embedding_type="positional",
    block_out_channels=[128,128,256,512],
    layers_per_block=2).to(DEVICE)

scheduler = DDPMScheduler(
    num_train_timesteps=TIMESTEPS,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon")

optimizer = torch.optim.Adam(Unet.parameters(),lr=LEARNING_RATE)
loss_term = nn.MSELoss()

## In training, we sample a batch of
## images, noise, and timesteps. The
## noisy image in passed to U-net to
## predict the noise and the loss is
## aimed at reducing the MSE between
## true noise and predicted noise.
for epoch in range(EPOCHS):
    Unet.train()
    pbar = tqdm(train_loader,desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for image, _ in pbar:
        image = image.to(DEVICE)
        noise = torch.randn_like(image).to(DEVICE)
        timesteps = torch.randint(0,TIMESTEPS,(image.shape[0],))
        timesteps = timesteps.to(DEVICE).long()
        noisyImgs = scheduler.add_noise(image, noise, timesteps)
        noisyPred = Unet(noisyImgs,timesteps).sample
        loss = loss_term(noise,noisyPred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

pipeline = DDPMPipeline(unet=Unet,scheduler=scheduler)
pipeline.to(DEVICE)
sampleData  = pipeline(batch_size=BATCH, generator=torch.manual_seed(0)).images

## Show the generated samples points
## Expecting good results, capturing
## low-frequency features (contents)
fig, axes = plt.subplots(8,8,figsize=(15,15))
for i, img in enumerate(sampleData):
    axes[i//8,i%8].imshow(img)
    axes[i//8,i%8].axis('off')