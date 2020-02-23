import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn import utils

from conditional_batchnorm import CategoricalConditionalBatchNorm2d


class MnistCNN(nn.Module):
    
    def __init__(self, n_channels, n_classes, n_filters):
        super(MnistCNN, self).__init__()
        
        def block(in_channels, out_channels,
                  stride=2, padding=1,
                  nonlinearity=nn.ReLU()):
            return [nn.Conv2d(in_channels, out_channels,
                        kernel_size=4, stride=stride, padding=padding),
                    nonlinearity]
            
        self.conv_blocks = nn.Sequential(
            *block(n_channels,  n_filters),
            *block(n_filters,   2*n_filters),
            *block(2*n_filters, 4*n_filters))
        
        self.fc = nn.Linear(4*n_filters*16, n_classes)
        
    def forward(self, x):
        h = self.conv_blocks(x)
        h = torch.flatten(h, start_dim=1)
        s = self.fc(h)
        return s


class DBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(DBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedDBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedDBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedDBlock(3, num_features)
        self.block2 = DBlock(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = DBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = DBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.l5 = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 8))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l5.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output
    


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class GBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(GBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))



class Generator(nn.Module):
    """Generator generates 32x32."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 8 * num_features * bottom_width ** 2)

        self.block2 = GBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = GBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = GBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


