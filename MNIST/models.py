## WGAN based Critic and Generator model. Adding
## spectral normalization. Will use trans/Conv2D
## and Batch normalization for models.  Test the
## model. Since the image are not so complicated
## will stick to small number of channels.

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary

class Critic(nn.Module):
    def __init__(self, in_channels, channelList, label, embedding, imsize):
        super().__init__()
        self.in_channels = in_channels
        self.embedding = embedding
        self.label = label
        self.imsize = imsize
        self.channelList = channelList

        assert self.embedding == self.imsize*self.imsize
        assert self.channelList[-1] == 1

        self.embed = nn.Embedding(self.label, self.embedding) ##[len=4096]
        self.inits = self._blocks(self.in_channels+1, self.channelList[0])
        layers = []
        for i in range(1, len(self.channelList)):
            layers.append(self._blocks(self.channelList[i-1], self.channelList[i]))
        self.layer = nn.ModuleList(layers)
    def _blocks(self, i, o):
        if o == 1:
            return spectral_norm(nn.Conv2d(i,o,4,2,1))
        return nn.Sequential(spectral_norm(nn.Conv2d(i,o,4,2,1)), nn.BatchNorm2d(o),nn.LeakyReLU(0.2))    
    def forward(self, x, y):
        shape = (-1, 1, self.imsize, self.imsize)
        input = torch.cat([x, torch.reshape(self.embed(y), shape)], dim=1)
        input = self.inits(input)
        for lays in self.layer:
            input = lays(input)
        return input ## Nx1x4x4

## x = torch.randn(10,1,64,64)
## y = torch.randint(0,10,(10,))
## critic = Critic(1,[2,4,8,16,32,1],10,4096,64)
## print(summary(critic, [x,y]))
################################################

class Generator(nn.Module):
    def __init__(self, z_dim, channelList, label, embedding):
        super().__init__()
        self.z_dim = z_dim
        self.label = label
        self.channelList = channelList
        self.embedding = embedding

        assert self.channelList[-1] == 1

        self.embed = nn.Embedding(self.label, self.embedding)
        self.inits = self._blocks(self.z_dim + self.embedding, self.channelList[0])
        layers = []
        for i in range(1, len(self.channelList)):
            layers.append(self._blocks(self.channelList[i-1], self.channelList[i]))
        self.layer = nn.ModuleList(layers)

    def _blocks(self, i, o):
        if o == 1:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1), nn.Tanh())
        return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1), nn.BatchNorm2d(o), nn.ReLU())
    
    def forward(self, z, y):
        input = torch.cat([z, torch.unsqueeze(torch.unsqueeze(self.embed(y), dim=2), dim=3)], dim=1)
        input = self.inits(input)
        for lays in self.layer:
            input = lays(input)
        return input##Nx1x64x64

## z = torch.randn(10,128,1,1)
## y = torch.randint(0,10,(10,))
## generator = Generator(128,[32,16,8,4,2,1],10,128)
## print(summary(generator, [z,y]))
################################################