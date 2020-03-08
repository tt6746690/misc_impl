import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class UpsampleConv(nn.Module):
    """ Upsample then Convolution. Better than ConvTranspose2d
            https://distill.pub/2016/deconv-checkerboard/
            https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor):
        super(UpsampleConv, self).__init__()
        self.scale_factor = scale_factor
        self.pad  = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)
        x = self.pad(x)
        x = self.conv(x)
        return x
    

def deconv3x3(in_planes, out_planes, stride=1, dilation=1):
    return UpsampleConv(in_planes, out_planes, kernel_size=3, stride=1, scale_factor=2)


def deconv1x1(in_planes, out_planes, stride=1, dilation=1):
    return UpsampleConv(in_planes, out_planes, kernel_size=1, stride=1, scale_factor=2)


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
        resample=None, norm_layer=nn.Identity, nonlinearity=nn.ReLU(inplace=True), resblk_1st=False):
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
        """
        super(ResidualBlock, self).__init__()
        
        if   resample == 'dn':
            residual_conv_resample = conv3x3(in_channels, out_channels, 2)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 2)
        elif resample == 'up':
            residual_conv_resample = deconv3x3(in_channels, out_channels)
            shortcut_conv_resample = deconv1x1(in_channels, out_channels)
        else:
            residual_conv_resample = conv3x3(in_channels, out_channels, 1)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 1)

        self.residual = nn.Sequential()
        self.residual.add_module('Normalization_1', norm_layer(in_channels))
        self.residual.add_module('Nonlinearity_1', nn.Identity() if resblk_1st else nonlinearity)
        self.residual.add_module('Conv_1', residual_conv_resample)
        self.residual.add_module('Normalization_2', norm_layer(out_channels))
        self.residual.add_module('Nonlinearity_2', nonlinearity)
        self.residual.add_module('Conv_2', conv3x3(out_channels, out_channels))
        
        if in_channels == out_channels and resample == None:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('Normalization_1', norm_layer(in_channels))
            self.shortcut.add_module('Conv_1', shortcut_conv_resample)
            
        
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
        nonlinearity = nn.ReLU(inplace=True)
        
        self.Linear = nn.Linear(dim_z, (self.bottom_width**2) * conv_channels[0])
        
        self.ResidualBlocks = nn.Sequential()
        for i in range(n_convs):
            upsample = conv_upsample[i]
            self.ResidualBlocks.add_module(f'ResBlock{"Up" if upsample else ""}_{i}',
                  ResidualBlock(conv_channels[i], conv_channels[i+1],
                                resample = "up" if upsample else None,
                                norm_layer = nn.BatchNorm2d,
                                nonlinearity = nonlinearity))
        
        self.NormalizationFinal = nn.BatchNorm2d(conv_channels[-1])
        self.NonlinearityFinal = nonlinearity
        self.ConvFinal = conv3x3(conv_channels[-1], im_channnels)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # 128
        x = self.Linear(x)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        # 1024x4x4
        x = self.ResidualBlocks(x)
        # 64x128x128
        x = self.NormalizationFinal(x)
        x = self.NonlinearityFinal(x)
        x = self.Tanh(self.ConvFinal(x))
        # 3x128x128
        return x

        
class Discriminator(nn.Module):
    
    def __init__(self, conv_channels = None, conv_dnsample = None):
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
        
        nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        
        self.ResidualBlocks = nn.Sequential()
        for i in range(n_convs):
            downsample = conv_dnsample[i]
            self.ResidualBlocks.add_module(f'ResBlock{"Dn" if downsample else ""}_{i}',
                  ResidualBlock(conv_channels[i], conv_channels[i+1],
                                resample = "dn" if downsample else None,
                                norm_layer = nn.Identity,
                                nonlinearity = nonlinearity,
                                resblk_1st = True if i == 0 else False))
        
        self.NonlinearityFinal = nonlinearity
        self.LinearFinal = nn.Linear(conv_channels[-1], 1)
        

    def forward(self, x):
        # 3x128x128
        x = self.ResidualBlocks(x)
        # 1024x4x4
        x = self.NonlinearityFinal(x)
        x = torch.sum(x, dim=(2,3))   # (global sum pooling)
        # 1024
        x = self.LinearFinal(x)
        # 1
        return x