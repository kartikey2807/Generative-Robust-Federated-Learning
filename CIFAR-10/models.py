## WGAN-GP implementation similar as CIFAR
## dataset. Main difference is that images
## are colored(3 channels) and complicated

from config import *
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

class _Block(nn.Module):
    def __init__(self, in_ch, out, trans = False, last=False):
        super().__init__()

        self.in_ch = in_ch
        self.out = out
        self.last = last
        self.trans = trans

        self.layer = None
        if not trans:
            self.layer = nn.Sequential(
                nn.Conv2d(self.in_ch,self.out,4,2,1),
                nn.LeakyReLU(0.2))
        else:
            if not last:
                self.layer = nn.Sequential(
                nn.ConvTranspose2d(self.in_ch,self.out,4,2,1),
                nn.BatchNorm2d(self.out),nn.ReLU())
            else:
                self.layer = nn.Sequential(
                nn.ConvTranspose2d(self.in_ch,self.out,4,2,1),
                nn.Tanh())
    def forward(self, x):
        return self.layer(x)
        
class Critic(nn.Module):
    def __init__(self, chn, label, embedding, imsize,in_ch=3):
        super().__init__()

        self.chn = chn
        self.label = label
        self.embedding = embedding
        self.imsize = imsize
        self.in_ch = in_ch
        '''
        Only assertion to check is that to
        check if embedding = pow(imsize,2)
        '''
        assert self.imsize == 32
        assert len(self.chn) == 3
        assert self.embedding == self.imsize*self.imsize
        self.layer = nn.ModuleList([
            _Block(self.in_ch+1,self.chn[0]),
            _Block(self.chn[0], self.chn[1]),
            _Block(self.chn[1], self.chn[2])])
        self.layer.append(nn.Flatten())
        self.layer.append(nn.Linear(self.chn[-1]*(self.imsize//8)*(self.imsize//8),1))
        ## self.layer.append(nn.LeakyReLU(0.2))
        ## self.layer.append(nn.Linear(256,1))
        self.embed = nn.Embedding(self.label,self.embedding)
    def forward(self, x, y):
        y = self.embed(y).view(-1,1,self.imsize,self.imsize)
        x = torch.cat([x,y],dim=1)
        for l in self.layer:
            x = l(x)
        return x ## BATCHx1

class Generator(nn.Module):
    def __init__(self, noise, gen_chn, label, embedding):
        super().__init__()

        self.noise = noise
        self.label = label
        self.gen_chn = gen_chn
        self.embedding = embedding

        assert len(self.gen_chn) == 4
        self.embed = nn.Embedding(self.label,self.embedding)
        self.fc_01 = nn.Sequential(
            nn.Linear(self.noise+self.embedding,4096),
            nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096,self.gen_chn[0]*2*2),
            nn.BatchNorm1d(self.gen_chn[0]*2*2), nn.ReLU())
        self.layer = nn.ModuleList([
            _Block(self.gen_chn[0],self.gen_chn[1],trans=True),
            _Block(self.gen_chn[1],self.gen_chn[2],trans=True),
            _Block(self.gen_chn[2],self.gen_chn[3],trans=True),
            _Block(self.gen_chn[3],3,trans=True,last=True)])
    def forward(self, z, y):
        input = torch.cat([z,self.embed(y)],dim=1)
        input = self.fc_01(input)
        input = input.view(-1,self.gen_chn[0],2,2)
        for l in self.layer:
            input = l(input)
        return input ## BATCHx3x32x32