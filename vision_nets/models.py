import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def apply_sn(module, use_sn):
    if use_sn:
        return spectral_norm(module)
    else:
        return module

def conv3x3(in_planes, out_planes, stride=1, dilation=1, use_sn=False):
    """3x3 convolution with padding"""
    module = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)
    return apply_sn(module, use_sn)


def conv1x1(in_planes, out_planes, stride=1, use_sn=False):
    """1x1 convolution"""
    module = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return apply_sn(module, use_sn)


class UpsampleConv(nn.Module):
    """ Upsample then Convolution. Better than ConvTranspose2d
            https://distill.pub/2016/deconv-checkerboard/
            https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor, use_sn=False):
        super(UpsampleConv, self).__init__()
        self.scale_factor = scale_factor
        self.pad  = torch.nn.ReflectionPad2d(kernel_size // 2)        
        self.conv = apply_sn(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride), use_sn)
        
    def forward(self, x):
        x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)
        x = self.pad(x)
        x = self.conv(x)
        return x
    

def deconv3x3(in_planes, out_planes, stride=1, dilation=1, use_sn=False):
    return UpsampleConv(in_planes, out_planes, kernel_size=3, stride=1, scale_factor=2, use_sn=use_sn)


def deconv1x1(in_planes, out_planes, stride=1, dilation=1, use_sn=False):
    return UpsampleConv(in_planes, out_planes, kernel_size=1, stride=1, scale_factor=2, use_sn=use_sn)


class ResidualBlock(nn.Module):
    """ Pre-activation Residual Block
            BN, nonlinearity, conv3x3, BN, nonlinearity, conv3x3

    References:
        https://arxiv.org/abs/1512.03385
        http://torch.ch/blog/2016/02/04/resnets.html

        ResBlock
            https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        WGAN-GP resnet architecture
            https://github.com/igul222/improved_wgan_training/blob/fa66c574a54c4916d27c55441d33753dcc78f6bc/gan_cifar_resnet.py#L159
            Generator: BN, ReLU, conv3x3, Tanh -> out
            PreactivationResblock: https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py
        SNGAN/Projection cGAN architecture
            https://github.com/pfnet-research/sngan_projection/blob/master/dis_models/resblocks.py
    """

    def __init__(self, in_channels, out_channels, 
                 resample=None,
                 norm_layer=nn.Identity,
                 nonlinearity=nn.ReLU(inplace=True),
                 resblk_1st=False,
                 use_sn=False):
        """
            resample
                \in {None, 'up', 'dn'}
            norm_layer
                \in {nn.Identity, nn.BatchNorm2d}
            nonlinearity
                either 
                    nn.ReLU(inplace=True)
                    nn.LeakyReLU(slope=0.2)
            resblk_1st
                if True, no nonlinearity before first `conv_1`
            use_sn
                Apply spectral normalization for each linear/conv layers
        """
        super(ResidualBlock, self).__init__()
        
        if   resample == 'dn':
            residual_conv_resample = conv3x3(in_channels, out_channels, 2, use_sn=use_sn)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 2, use_sn=use_sn)
        elif resample == 'up':
            residual_conv_resample = deconv3x3(in_channels, out_channels, use_sn=use_sn)
            shortcut_conv_resample = deconv1x1(in_channels, out_channels, use_sn=use_sn)
        else:
            residual_conv_resample = conv3x3(in_channels, out_channels, 1, use_sn=use_sn)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 1, use_sn=use_sn)

        self.residual = nn.Sequential()
        self.residual.add_module('normalization_1', norm_layer(in_channels))
        self.residual.add_module('nonlinearity_1', nn.Identity() if resblk_1st else nonlinearity)
        self.residual.add_module('conv_1', residual_conv_resample)
        self.residual.add_module('normalization_2', norm_layer(out_channels))
        self.residual.add_module('nonlinearity_2', nonlinearity)
        self.residual.add_module('conv_2', conv3x3(out_channels, out_channels, use_sn=use_sn))
        
        if in_channels == out_channels and resample == None:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('normalization_1', norm_layer(in_channels))
            self.shortcut.add_module('conv_1', shortcut_conv_resample)
            
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
    
class Generator(nn.Module):
    
    def __init__(self, conv_channels = None, conv_upsample = None, dim_z = 128, im_channnels = 3):
        """
            conv_channels
                [1024, 1024, 512, 256, 128, 64]
                     c1    c2   c3   c4   c5
            conv_upsample
                4x4 -> 128x128    [True, True, True, True, True]
                4x4 -> 64x64      [True, True, True, True]
                4x4 -> 32x32      [True, True, True]
            im_channnels
                3 for color image
                1 for grayscale image
        """
        super(Generator, self).__init__()
        
        n_convs = len(conv_channels) - 1
        assert(n_convs > 0)
        assert(n_convs == len(conv_upsample))
        
        self.bottom_width = 4
        self.nonlinearity = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(dim_z, (self.bottom_width**2) * conv_channels[0])
        
        self.residual_blocks = nn.Sequential()
        for i in range(n_convs):
            upsample = conv_upsample[i]
            self.residual_blocks.add_module(f'residual_block{"_up" if upsample else ""}_{i}',
                  ResidualBlock(conv_channels[i], conv_channels[i+1],
                                resample = "up" if upsample else None,
                                norm_layer = nn.BatchNorm2d,
                                nonlinearity = self.nonlinearity))
        
        self.normalization_final = nn.BatchNorm2d(conv_channels[-1])
        self.conv_final = conv3x3(conv_channels[-1], im_channnels)
        self.nonlinearity_final = nn.Tanh()

    def forward(self, x):
        # 128
        x = self.linear(x)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        # 1024x4x4
        x = self.residual_blocks(x)
        # 64x128x128
        x = self.normalization_final(x)
        x = self.nonlinearity(x)
        x = self.conv_final(x)
        x = self.nonlinearity_final(x)
        # 3x128x128
        return x

        
class Discriminator(nn.Module):
    
    def __init__(self, conv_channels = None, conv_dnsample = None, use_sn=True):
        """
            conv_channels
                [3, 64, 128, 256, 512, 1024, 1024]
                  c1  c2   c3   c4   c5    c6
            conv_dnsample
                128x128 -> 4x4    [True, True, True, True, True, False]
                64x64 -> 4x4      [True, True, True, True]
                32x32 -> 4x4      [True, True, True]
                
        Projection cGAN 
            conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
            conv_dnsample = [True, True, True, True, True, False]
        """
        super(Discriminator, self).__init__()
        
        n_convs = len(conv_channels) - 1
        assert(n_convs > 0)
        assert(n_convs == len(conv_dnsample))
        
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual_blocks = nn.Sequential()
        for i in range(n_convs):
            downsample = conv_dnsample[i]
            self.residual_blocks.add_module(f'residual_block{"_dn" if downsample else ""}_{i}',
                  ResidualBlock(conv_channels[i], conv_channels[i+1],
                                resample = "dn" if downsample else None,
                                norm_layer = nn.Identity,
                                nonlinearity = self.nonlinearity,
                                resblk_1st = True if i == 0 else False,
                                use_sn = use_sn))

        self.linear = apply_sn(nn.Linear(conv_channels[-1], 1), use_sn)

    def forward(self, x):
        # 3x128x128
        x = self.residual_blocks(x)
        # 1024x4x4
        x = self.nonlinearity(x)
        x = torch.sum(x, dim=(2,3))   # (global sum pooling)
        # 1024
        x = self.linear(x)
        # 1
        return x