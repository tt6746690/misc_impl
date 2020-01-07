# reference: https://github.com/pytorch/examples/blob/master/dcgan/main.py
#
import os
import itertools
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision 
import torchvision.datasets as datasets



class Generator(nn.Module):

    def __init__(self, nz, nf, nc):
        """
            nz      dimension of noise 
            nf      dimension of features in last conv layer
            nc      number of channels in the image

            In DCGAN paper for LSUN dataset, nz=100, nf=128, nc=3
        """
        super(Generator, self).__init__()

        def block(in_channels, out_channels, stride=2, padding=1, batch_norm=True, nonlinearity=nn.ReLU(True)):
            """ stride=1, padding=0: H_out = H_in + 3       # 1 -> 4
                stride=2, padding=1: H_out = 2 * H_in       # doubles
            """
            return [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity,
            ]

        self.model = nn.Sequential(
            # (nz)   x 1 x 1
            *block(nz,   8*nf, stride=1, padding=0),
            # (8*nf) x 4 x 4
            *block(8*nf, 4*nf),
            # (4*nf) x 8 x 8
            *block(4*nf, 2*nf),
            # (2*nf) x 16 x 16
            *block(2*nf,   nf),
            # (nf) x 32 x 32
            *block(nf,     nc, batch_norm=False, nonlinearity=nn.Tanh()),
            # (nc) x 64 x 64
        )

    def forward(self, z):
        """
            z       (N, nz, 1, 1)
                noise vector
            Returns (N, nc, h, w)
                image generated from model distribution
                
        """
        return self.model(z)
    
    
class Discriminator(nn.Module):
    
    def __init__(self, nc, nf):
        """
            nc      number of channels in the image
            nf      dimension of features of first conv layer

            In DCGAN paper for LSUN dataset, nc=3
        """
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels,
                  stride=2, padding=1,
                  batch_norm=True,
                  nonlinearity=nn.LeakyReLU(0.2, inplace=True)):
            """ stride=1, padding=0: H_out = H_in - 3              # 4 -> 1
                stride=2, padding=1: H_out = floor((H_in-1)/2 +1)  # roughly halves
            """
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity,
            ]
        
        self.model = nn.Sequential(
            # (nc) x 64 x 64
            *block(nc,     nf, batch_norm=False),
            # (nf) x 32 x 32
            *block(nf,   2*nf),
            # (2*nf) x 16 x 16
            *block(2*nf, 4*nf),
            # (4*nf) x 8 x 8
            *block(4*nf, 8*nf),
            # (8*nf) x 4 x 4
            *block(8*nf, 1, stride=1, padding=0, batch_norm=False, nonlinearity=nn.Sigmoid()),
            # 1 x 1 x 1
        )
        
        
    def forward(self, x):
        """
            x        (N, nc, h, w)
            Returns  (N,)
                classification probability that x comes from data distribution
        """
        x = self.model(x)
        return  x.view(-1, 1).squeeze(1)
        