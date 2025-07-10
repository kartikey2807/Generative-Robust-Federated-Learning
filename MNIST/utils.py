## Contains dataset and weight initializations
## MNIST Dataset is used with minor transforms
## eval(....) computes  loss during validation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from config import *
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

t = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    ## transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
    ])
"""
Take the image, rotate it by (-10,+10) degrees
and normalize about 0.5 mean and 0.5 std. Next
create a function to initialize weights in all
of layer to 0 mean and 0.02 standard deviation
"""

def eval(x, y, z, critic, generator):
    fakeImage = generator(z,y)
    real_loss = torch.mean(critic(x,y))
    fake_loss = torch.mean(critic(fakeImage,y))
    test_loss = torch.mean(real_loss) - torch.mean(fake_loss)
    return test_loss.cpu().detach().numpy()

train_dataset = torchvision.datasets.MNIST('MNIST/dataset/', train=True, transform=t, download=True)
def weight_initialization(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.LazyBatchNorm2d)):
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)