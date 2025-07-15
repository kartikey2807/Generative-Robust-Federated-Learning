## Apply conditional WGAN-GP model for toy
## dataset. Since the number of feature is
## less, the models will not be deep. make
## MLPs for critic and generator.

import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn.utils import spectral_norm
from torch.optim import Adam

class Critic(nn.Module):
    def __init__(self, input, hidden, embedding, label):
        super().__init__()
        
        self.input = input
        self.hidden = hidden
        self.embedding = embedding
        self.label = label
        self.embed = nn.Embedding(self.label, self.embedding)
        
        self.layer = nn.Sequential(
            self._block(self.input+self.embedding, self.hidden[0]),
            self._block(self.hidden[0],self.hidden[1]),
            nn.Linear(self.hidden[1],1))
        
    def _block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features,out_features,bias=False), nn.LeakyReLU(0.2))
    
    def forward(self, x, y):
        y_embed = self.embed(y).to(dtype=x.dtype, device=x.device)
        return self.layer(torch.cat([x,y_embed],dim=1))

class Generator(nn.Module):
    def __init__(self, z_dim, gendim, embedding, label):
        super().__init__()

        '''
        z_dim: noise dimension
        gendim: Generator hidden layer//SHIT NAME (μ_μ)
        '''

        self.z_dim = z_dim
        self.gendim = gendim
        self.embedding = embedding
        self.label = label
        self.embed = nn.Embedding(self.label, self.embedding)

        self.layer = nn.Sequential(
            self._block(self.z_dim+self.embedding, self.gendim[0]),
            self._block(self.gendim[0], self.gendim[1]),
            self._block(self.gendim[1], self.gendim[2]),
            nn.Linear(self.gendim[2],2),nn.Tanh()) ##[-1,1]

    def _block(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features), nn.ReLU())
    
    def forward(self, z, y):
        y_embed = self.embed(y).to(dtype=z.dtype, device=z.device)
        return self.layer(torch.cat([z,y_embed], dim=1))