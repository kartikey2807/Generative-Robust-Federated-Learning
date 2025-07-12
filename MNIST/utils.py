## Contains weight initialization for MNIST
## dataset is and has minor transformations
## eval(..) computes loss during validation
## Also, gradient penalty is implemented.

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

OPTIMA = 10000
def eval(x, y, z, critic, generator):
    global OPTIMA
    fakeImage = generator(z,y)
    real_loss = torch.mean(critic(x,y))
    fake_loss = torch.mean(critic(fakeImage,y))
    test_loss = torch.mean(real_loss) - torch.mean(fake_loss)
    test_loss = test_loss.cpu().detach().numpy()
    if test_loss <= OPTIMA:
        OPTIMA = test_loss
        torch.save(generator.state_dict(), 'model.pth')
    return test_loss

train_dataset = torchvision.datasets.MNIST('MNIST/dataset/', train=True, transform=t, download=True)
def weight_initialization(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.LazyBatchNorm2d)):
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

def gradient_penalty(critic, real, fake, label, device):
    BATCH,C,H,W = real.shape
    alpha = torch.rand((BATCH,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_image = (alpha*real) + ((1-alpha)*fake)

    '''
    Pass the interpolated image through critic
    and force the gradient wrt. input to be as
    close to 1 as possible. Lipschitz statisfy
    '''
    mixed_loss = critic(interpolated_image, label)
    gradient_1 = torch.autograd.grad(
        outputs=mixed_loss,
        inputs=interpolated_image,
        grad_outputs=torch.ones_like(mixed_loss),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient_1 = gradient_1.view(gradient_1.shape[0],-1)
    gradient_norm = gradient_1.norm(2, dim=1)
    return torch.mean((gradient_norm - 1)**2)