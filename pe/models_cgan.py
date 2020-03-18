import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    """ https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    """
    
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, c):
        out = self.bn(x)
        gamma, beta = self.embed(c.view(-1)).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


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
                 nonlinearity=nn.ReLU(),
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
        
        self.identity_shortcut = (in_channels == out_channels and resample == None)
        
        if   resample == 'dn':
            residual_conv_resample = conv3x3(in_channels, out_channels, 2, use_sn=use_sn)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 2, use_sn=use_sn)
        elif resample == 'up':
            residual_conv_resample = deconv3x3(in_channels, out_channels, use_sn=use_sn)
            shortcut_conv_resample = deconv1x1(in_channels, out_channels, use_sn=use_sn)
        else:
            residual_conv_resample = conv3x3(in_channels, out_channels, 1, use_sn=use_sn)
            shortcut_conv_resample = conv1x1(in_channels, out_channels, 1, use_sn=use_sn)

        self.residual_normalization_1 = norm_layer(in_channels)
        self.residual_nonlinearity_1 = nn.Identity() if resblk_1st else nonlinearity
        self.residual_conv_1 = residual_conv_resample
        self.residual_normalization_2 = norm_layer(out_channels)
        self.residual_nonlinearity_2 = nonlinearity
        self.residual_conv_2 = conv3x3(out_channels, out_channels, use_sn=use_sn)
        
        if not self.identity_shortcut:
            self.shortcut_normalization_1 = norm_layer(in_channels)
            self.shortcut_conv_1 = shortcut_conv_resample

    def forward(self, x):

        identity = x
        
        if self.identity_shortcut:
            s = identity
        else:
            s = self.shortcut_normalization_1(identity)
            s = self.shortcut_conv_1(s)
            
        x = self.residual_normalization_1(x)
        x = self.residual_nonlinearity_1(x)
        x = self.residual_conv_1(x)
        x = self.residual_normalization_2(x)
        x = self.residual_nonlinearity_2(x)
        x = self.residual_conv_2(x)
        
        return x + s 
    
    
class ConditionalResidualBlock(ResidualBlock):
    """ Residual block w/ categorical conditional BatchNorm2d
    """
    
    def __init__(self, in_channels, out_channels,
                 resample=None,
                 norm_layer=nn.BatchNorm2d,
                 nonlinearity=nn.ReLU(inplace=True),
                 resblk_1st=False,
                 use_sn=False):
        """
            norm_layer
                initialize w/ num_features
        """
        
        super(ConditionalResidualBlock, self).__init__(
            in_channels, out_channels,
            resample = resample,
            norm_layer = norm_layer,
            nonlinearity = nonlinearity,
            resblk_1st = resblk_1st,
            use_sn = use_sn)
        
    def forward(self, x, c):
        
        identity = x
        
        if self.identity_shortcut:
            s = identity
        else:
            s = self.shortcut_normalization_1(identity, c)
            s = self.shortcut_conv_1(s)
            
        x = self.residual_normalization_1(x, c)
        x = self.residual_nonlinearity_1(x)
        x = self.residual_conv_1(x)
        x = self.residual_normalization_2(x, c)
        x = self.residual_nonlinearity_2(x)
        x = self.residual_conv_2(x)
        
        return x + s
        
        
class Generator(nn.Module):
    
    def __init__(self, conv_channels, conv_upsample,
                 resblk_cls = ResidualBlock,
                 norm_layer = nn.BatchNorm2d,
                 dim_z = 128,
                 im_channels = 3):
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
            num_classes
                if not None, use conditional batchnorm
        """
        super(Generator, self).__init__()
        
        n_convs = len(conv_channels) - 1
        assert(n_convs > 0)
        assert(n_convs == len(conv_upsample))
        
        self.n_convs = n_convs
        self.bottom_width = 4
        self.nonlinearity = nn.ReLU()
        
        self.linear = nn.Linear(dim_z, (self.bottom_width**2) * conv_channels[0])
        
        for i in range(n_convs):
            upsample = conv_upsample[i]
            self.add_module(
                f'residual_block_{i}',
                resblk_cls(conv_channels[i], conv_channels[i+1],
                           resample = "up" if upsample else None,
                           norm_layer = norm_layer,
                           nonlinearity = self.nonlinearity))
        
        self.normalization_final = norm_layer(conv_channels[-1])
        self.conv_final = conv3x3(conv_channels[-1], im_channels)
        self.nonlinearity_final = nn.Tanh()

    def forward(self, x):
        # bottom_width = 4
        # conv_channels = [1024, 1024, 512, 256, 128, 64]
        # conv_upsample = [True, True, True, True, True]
        # im_channnels = 3
        #
        # 128
        x = self.linear(x)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        # 1024x4x4
        for i in range(self.n_convs):
            x = getattr(self, f'residual_block_{i}')(x)
        # 64x128x128
        x = self.normalization_final(x)
        x = self.nonlinearity(x)
        x = self.conv_final(x)
        x = self.nonlinearity_final(x)
        # 3x128x128
        return x
    
    
class ConditionalGenerator(Generator):
    
    def __init__(self, conv_channels, conv_upsample, num_classes,
                 dim_z = 128,
                 im_channels = 3):
        """
            norm_layer = lambda num_features: ConditionalBatchNorm2d(num_features, num_classes)
        """ 
        resblk_cls = ConditionalResidualBlock
        norm_layer = lambda num_features: ConditionalBatchNorm2d(num_features, num_classes)
        
        super(ConditionalGenerator, self).__init__(conv_channels, conv_upsample,
            resblk_cls = resblk_cls,
            norm_layer = norm_layer,
            dim_z = dim_z,
            im_channels = im_channels)

    def forward(self, x, c):
        """ x    batch_size x im_channels x h x w
            c    batch_size
        """
        c = c.view(-1)
        #
        # 128
        x = self.linear(x)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        # 1024x4x4
        for i in range(self.n_convs):
            x = getattr(self, f'residual_block_{i}')(x, c)
        # 64x128x128
        x = self.normalization_final(x, c)
        x = self.nonlinearity(x)
        x = self.conv_final(x)
        x = self.nonlinearity_final(x)
        # 3x128x128
        return x
    

class Discriminator(nn.Module):
    
    def __init__(self, conv_channels, conv_dnsample, use_sn=True):
        """
            conv_channels
                [3, 64, 128, 256, 512, 1024, 1024]
                  c1  c2   c3   c4   c5    c6
            conv_dnsample
                128x128 -> 4x4    [True, True, True, True, True, False]
                64x64 -> 4x4      [True, True, True, True]
                32x32 -> 4x4      [True, True, True]
        """
        super(Discriminator, self).__init__()
        
        n_convs = len(conv_channels) - 1
        assert(n_convs > 0)
        assert(n_convs == len(conv_dnsample))
        
        self.nonlinearity = nn.LeakyReLU(0.2)

        self.residual_blocks = nn.Sequential()
        for i in range(n_convs):
            downsample = conv_dnsample[i]
            self.residual_blocks.add_module(f'residual_block_{i}',
                  ResidualBlock(conv_channels[i], conv_channels[i+1],
                                resample = "dn" if downsample else None,
                                norm_layer = nn.Identity,
                                nonlinearity = self.nonlinearity,
                                resblk_1st = True if i == 0 else False,
                                use_sn = use_sn))

        self.linear = apply_sn(nn.Linear(conv_channels[-1], 1), use_sn)

    def forward(self, x):
        # conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
        # conv_dnsample = [True, True, True, True, True]
        #
        # 3x128x128
        x = self.residual_blocks(x)
        # 1024x4x4
        x = self.nonlinearity(x)
        x = torch.sum(x, dim=(2,3))   # (global sum pooling)
        # 1024
        x = self.linear(x)
        # 1
        return x
    
    
class ConditionalDiscriminator(Discriminator):
    
    def __init__(self, conv_channels, conv_dnsample, num_classes, use_sn=True):
        """
            Projection cGAN (ImageNet)
                conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
                conv_dnsample = [True, True, True, True, True, False]
        """
        super(ConditionalDiscriminator, self).__init__(conv_channels, conv_dnsample, use_sn=use_sn)
        
        self.c_embed = apply_sn(nn.Embedding(num_classes, conv_channels[-1]), use_sn)
        
    def forward(self, x, c):
        """ x    batch_size x im_channels x h x w
            c    batch_size
        """
        c = c.view(-1)
        # conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
        # conv_dnsample = [True, True, True, True, True]
        #
        # 3x128x128
        x = self.residual_blocks(x)
        # 1024x4x4
        x = self.nonlinearity(x)
        x = torch.sum(x, dim=(2,3))   # (global sum pooling)
        # 1024
        
        # sigmoid^-1(p(real/fake|x,c)) =
        #     log(p_data(x)/p_model(x)) + 
        #     log(p_data(c|x)/p_model(c|x))
        x = self.linear(x) + \
            torch.sum(self.c_embed(c) * x, dim=1, keepdim=True)
        # 1
        return x
        



class OrdinalConditionalDiscriminator(Discriminator):
    """ conditional discriminator where the conditioned variable is ordinal
    
        Taken from:
            https://github.com/batmanlab/Explanation_by_Progressive_Exaggeration/blob/master/src/explainer.py
    """
    
    def __init__(self, conv_channels, conv_dnsample, num_classes, use_sn=True):
        """
            Projection cGAN (ImageNet)
                conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
                conv_dnsample = [True, True, True, True, True, False]
        """
        super(OrdinalConditionalDiscriminator, self).__init__(conv_channels, conv_dnsample, use_sn=use_sn)
        
        self.c_embed = apply_sn(nn.Embedding(num_classes, conv_channels[-1]), use_sn)
        
    def forward(self, x, c):
        """ x    batch_size x im_channels x h x w
            c    batch_size
        """
        c = c.view(-1)
        # conv_channels = [3, 64, 128, 256, 512, 1024, 1024]
        # conv_dnsample = [True, True, True, True, True]
        #
        # 3x128x128
        x = self.residual_blocks(x)
        # 1024x4x4
        x = self.nonlinearity(x)
        x = torch.sum(x, dim=(2,3))   # (global sum pooling)
        # 1024
        
        # sigmoid^-1(p(real/fake|x,c)) =
        #     log(p_data(x)/p_model(x)) + 
        #     log(p_data(c|x)/p_model(c|x))
        all_classes = torch.arange(0, num_classes, dtype=torch.long, device=x.device)
        W = x @ self.c_embed(all_classes).T
        W = torch.cumsum(W, dim=1)
        # 10
        f_1 = W.gather(dim=1, index=c.view(-1,1))
        f_2 = self.linear(x)
        
        x = f_1 + f_2
        # 1
        return x
    

class ConditionalAutoencoder(nn.Module):
    
    def __init__(self, enc_channels, dec_channels, num_classes,
                 dim_z = 128,
                 im_channels = 3):
        """
            enc_channels
                [64, 128, 256, 256]
                   c1   c2   c3
            dec_channels
                [256, 128, 64, 64]
                   c1   c2   c3
            num_classes
                if not None, use conditional batchnorm
        """
        super(ConditionalAutoencoder, self).__init__()
        
        n_enc_blks = len(enc_channels) - 1
        n_dec_blks = len(dec_channels) - 1
        assert(n_enc_blks > 0)
        assert(n_dec_blks > 0)
        
        self.n_enc_blks = n_enc_blks
        self.n_dec_blks = n_dec_blks
        self.bottom_width = 4
        self.nonlinearity = nn.ReLU()
        
        resblk_cls = ConditionalResidualBlock
        norm_layer = lambda num_features: ConditionalBatchNorm2d(num_features, num_classes)
        
        self.normalization_initial = norm_layer(im_channels)
        self.conv_initial = conv3x3(im_channels, enc_channels[0])
        
        for i in range(n_enc_blks):
            self.add_module(
                f'residual_block_enc_{i}',
                resblk_cls(enc_channels[i], enc_channels[i+1],
                           resample = "dn",
                           norm_layer = norm_layer,
                           nonlinearity = self.nonlinearity,
                           resblk_1st = True if i == 0 else False))
        
        for i in range(n_dec_blks):
            self.add_module(
                f'residual_block_dec_{i}',
                resblk_cls(dec_channels[i], dec_channels[i+1],
                           resample = "up",
                           norm_layer = norm_layer,
                           nonlinearity = self.nonlinearity))
            
        self.normalization_final = norm_layer(dec_channels[-1])
        self.conv_final = conv3x3(dec_channels[-1], im_channels)
        self.nonlinearity_final = nn.Tanh()

    def forward(self, x, c):
        """ x    batch_size x im_channels x h x w
            c    batch_size
            Returns  
                 batch_size x im_channels x h x w
        """
        c = c.view(-1)
        # bottom_width = 4
        # enc_channels = [64, 128, 256, 256]
        # dec_channels = [256, 128, 64, 64]
        # im_channnels = 3
        #
        # 3x32x32
        x = self.normalization_initial(x, c)
        x = self.nonlinearity(x)
        x = self.conv_initial(x)
        # 64x32x32
        for i in range(self.n_enc_blks):
            x = getattr(self, f'residual_block_enc_{i}')(x, c)
        # 256x4x4
        z = x
        # 256x4x4
        for i in range(self.n_dec_blks):
            x = getattr(self, f'residual_block_dec_{i}')(x, c)
        # 64x32x32
        x = self.normalization_final(x, c)
        x = self.nonlinearity(x)
        x = self.conv_final(x)
        x = self.nonlinearity_final(x)
        # 3x32x32
        return x, z