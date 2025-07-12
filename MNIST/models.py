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
    def __init__(self, inp_ch, chList, label, embedding, imsize):
        super().__init__()
        self.inp_ch = inp_ch
        self.chList = chList
        self.label  = label
        self.imsize = imsize
        self.embedding = embedding

        assert self.embedding == self.imsize*self.imsize

        self.embed = nn.Embedding(self.label,self.embedding)
        self.layer = nn.Sequential(
            self._blocks(self.inp_ch+1, self.chList[0]),
            self._blocks(self.chList[0],self.chList[1]),
            self._blocks(self.chList[1],self.chList[2]),
            self._blocks(self.chList[2],self.chList[3]),
            nn.Flatten(),
            nn.Linear(self.chList[3]*4*4, 1))
        
    def _blocks(self, i, o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),nn.LeakyReLU(0.2))
    
    def forward(self, x, y):
        shape = (-1,1,self.imsize,self.imsize)
        return self.layer(torch.cat([x,torch.reshape(self.embed(y), shape)], dim=1))

class Generator(nn.Module):
    def __init__(self, z_dim, chList, label, embedding):
        super().__init__()
        self.z_dim = z_dim
        self.label = label
        self.chList = chList
        self.embedding = embedding

        self.embed = nn.Embedding(self.label, self.embedding)
        self.layer = nn.Sequential(
            self._blocks(self.chList[0],self.chList[1]),
            self._blocks(self.chList[1],self.chList[2]),
            self._blocks(self.chList[2],self.chList[3]),
            self._blocks(self.chList[3],1))
        self.fc_01 = nn.Linear(self.z_dim+self.embedding,self.chList[0]*4*4)
    
    def _blocks(self, i, o):
        if o == 1:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.Tanh())
        else:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.BatchNorm2d(o), nn.ReLU())
    
    def forward(self, z, y):
        input = torch.cat([z,self.embed(y)], dim=1)
        input = self.fc_01(input)
        input = torch.reshape(input,(-1,self.chList[0], 4, 4))
        return self.layer(input)