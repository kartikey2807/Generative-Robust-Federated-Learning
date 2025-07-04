## Implement a simple Wasserstein GAN model working with
## toy-dataset. Since the feature space is small, models
## will not be very deep. Use MLP for both generator and
## discriminator. Use Wasserstein loss as objective. And
## apply spectral normalization for satisfying Lipschitz
## constraint. Make it conditional

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
            self._block(self.hidden[0], self.hidden[1]),
            spectral_norm(nn.Linear(self.hidden[1],1)))
        
    def _block(self, in_features, out_features):
        return nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features, bias=False)),
            nn.LeakyReLU(0.2))
    def forward(self, x, y):
        y_embed = self.embed(y).to(dtype=x.dtype, device=x.device)
        return self.layer(torch.cat([x,y_embed], dim=1))

## VALIDATION: create a dummy model and check if outputs
## align Count the number of parameters.
## test = torch.randn(10,2)
## test_label = torch.randint(0, 2, (10,))
## critic = Critic(2, [8,4], 2, 2)
## print(summary(critic, [test,test_label]))
########################################################

class Generator(nn.Module):
    def __init__(self, z_dim, gendim, embedding, label):
        super().__init__()
        ## z_dim: noise dimension
        ## gendim: number of hidden feature in generator
        self.z_dim = z_dim
        self.gendim = gendim
        self.embedding = embedding
        self.label = label
        self.embed = nn.Embedding(self.label, self.embedding)

        self.scale = torch.tensor(1.5)
        self.layer = nn.Sequential(
            self._block(self.z_dim+self.embedding, self.gendim[0]),
            self._block(self.gendim[0], self.gendim[1]),
            nn.Linear(self.gendim[1],2), nn.Tanh())
            ## Tanh() keeps values bounded b/w -1 and 1

    def _block(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features,out_features,bias=False),
            nn.BatchNorm1d(out_features), nn.ReLU())
    def forward(self, z, y):
        y_embed = self.embed(y).to(dtype=z.dtype, device=z.device)
        return self.layer(torch.cat([z,y_embed], dim=1))

## VALIDATION: generate samples from gaussian noise, and
## labels. Check the output on dummy model
## noise = torch.randn(10,2)
## labels = torch.randint(0, 2, (10,))
## gen = Generator(2, [2,2], 2, 2)
## print(summary(gen, input_data=[noise,labels]))
########################################################