# flaxmodels: https://github.com/matthias-wright/flaxmodels
# flaxvision: https://github.com/rolandgvc/flaxvision/blob/master/flaxvision/models/resnet.py
#     mimics torchvision ... has pretrained options ...
# flax/examples: https://github.com/google/flax/blob/main/examples/imagenet/models.py
#     most succint but no loading of pretrained weights
#
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as np

ModuleDef = Any


class NormIdentity(nn.Module):
    scale_init: Callable = None

    @nn.compact
    def __call__(self, X, **kwargs):
        return X


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = np.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        # (bsz, 224, 224, 1)
        x = conv(self.num_filters, (7, 7), (2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        # (bsz, 112, 112, 64)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        # (bsz, 56, 56, 64)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
        # (bsz, 7, 7, 512)
        x = np.mean(x, axis=(1, 2))
        # (bsz, 512)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        # (bsz, num_classes)
        x = np.asarray(x, self.dtype)
        return x


class ResNetTrunk(nn.Module):
    """ResNetV1. Trunk """
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = np.float32
    act: Callable = nn.relu
    disable_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)
        if self.disable_bn:
            norm = NormIdentity
        # resnet50
        # (bsz, 224, 224, 1)
        x = conv(self.num_filters, (7, 7), (2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        # (bsz, 112, 112, 64)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        # (bsz, 56, 56, 64)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # strides=[1, 2, 2, ...] for conv3x3 in `block_cls`
                # stride can be seen as downsample scale/rate
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
            # i=0, (bsz, 56, 56,  256)
            # i=1, (bsz, 28, 28,  512)
            # i=2, (bsz, 14, 14, 1024)
            # i=3, (bsz,  7,  7, 2048)
        return x


ResNet18Trunk = partial(ResNetTrunk, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock)
ResNet34Trunk = partial(ResNetTrunk, stage_sizes=[3, 4, 6, 3],
                        block_cls=ResNetBlock)
ResNet50Trunk = partial(ResNetTrunk, stage_sizes=[3, 4, 6, 3],
                        block_cls=BottleneckResNetBlock)

ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)

ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)


# BagNet
#
# References
# pytorchcv: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/bagnet.py
# rebias: https://github.com/clovaai/rebias/blob/master/models/imagenet_models.py (basicblock modification)
# imgclsmob(torch): https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/bagnet.py
#
#
# What does bagnet do ?
# In resnet50, 1 BottleneckBlock has
#             [conv1x1, conv3x3(stride=1,padding=1), conv1x1]
#                 for all blocks in first stage
#             [conv1x1, conv3x3(stride=2,padding=1), conv1x1]
#                 for all blocks in subsequent stage
# In bagnet
#     `kernel3` is the number of `conv3x3`s in each stage of resnet
#     bagnet essentially replace conv3x3 with conv1x1s
#        initial_conv has receptive field of (7, 7)
#        any subsequent conv3x3 increase receptive field
#            by 1 when stride=1
#            by
#
#     if kernel3=1,
#         BottleneckBlock has
#             [conv1x1, conv3x3(stride=2), conv1x1]
#                 for first block in each stage
#             [conv1x1, conv1x1(stride=1), conv1x1]
#                 for subsequent block in each stage
#     if kernel3=0,
#         BottleneckBlock has
#             [conv1x1, conv1x1(stride=2), conv1x1]
#                 for all block in each stage ...
#
# Note, conv3x3->conv1x1 just need to change padding=1->padding=0
#     and the downsampling rate maintains the same
#
#        ```
#         for ksize, stride, padding in [(3, 1, 1),
#                                        (3, 2, 1),
#                                        (1, 1, 0),
#                                        (1, 2, 0)]:
#             x = random.normal(key, (112, 112, 64))
#             m = nn.Conv(64, kernel_size=(ksize, ksize),
#                             strides=(stride, stride),
#                             padding=[(padding,padding), (padding,padding)])
#             params = m.init(key, x)
#             m = m.bind(params)
#             x = m(x)
#             print(f'(ksize={ksize}, stride={stride}, padding={padding}):\t{x.shape}')
#
#         >>>> (ksize=3, stride=1, padding=1):	(112, 112, 64)
#         >>>> (ksize=3, stride=2, padding=1):	(56, 56, 64)
#         >>>> (ksize=1, stride=1, padding=0):	(112, 112, 64)
#         >>>> (ksize=1, stride=2, padding=0):	(56, 56, 64)
#        ```


class BagNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    conv2_ksize: int = 1

    @property
    def ksize(self):
        return (self.conv2_ksize, self.conv2_ksize)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, self.ksize, self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckBagNetBlock(nn.Module):
    """Bottleneck BagNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    conv2_ksize: int = 1

    @property
    def ksize(self):
        return (self.conv2_ksize, self.conv2_ksize)

    @nn.compact
    def __call__(self, x):

        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, self.ksize, self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BagNetTrunk(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_conv3x3_per_stage: Sequence[int]
    num_filters: int = 64
    dtype: Any = np.float32
    act: Callable = nn.relu
    disable_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)
        if self.disable_bn:
            norm = NormIdentity

        # (bsz, 224, 224, 1)
        x = conv(self.num_filters, (1, 1), (1, 1))(x)
        x = conv(self.num_filters, (3, 3), (1, 1))(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        # (bsz, 224, 224, 64)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        # (bsz, 112, 112, 64)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # strides=[1, 2, 2, ...] for conv3x3 in `Bottleneck`
                #     stride can be seen as downsample scale/rate
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                # conv2_ksize for conv3x3 in `Bottleneck`
                #     first `num_conv3x3` are conv3x3
                num_conv3x3 = self.num_conv3x3_per_stage[i]
                conv2_ksize = 3 if j < num_conv3x3 else 1

                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act,
                                   conv2_ksize=conv2_ksize)(x)

            # i=0 (bsz, 112, 112,  256)
            # i=1 (bsz,  56,  56,  512)
            # i=2 (bsz,  28,  28, 1024)
            # i=3 (bsz,  14,  14, 2048)
        return x

    @property
    def receptive_field(self):
        filtered = list(filter(lambda x: x[0] == self.num_conv3x3_per_stage,
                               bagnet_num_conv3x3_and_receptive_fields))
        if len(filtered) != 1:
            raise ValueError(f'{self.num_conv3x3_per_stage} not supported')
        else:
            return filtered[0][1]


class BagNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_conv3x3_per_stage: Sequence[int]
    num_classes: int
    num_filters: int = 64
    dtype: Any = np.float32
    act: Callable = nn.relu
    disable_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)
        if self.disable_bn:
            norm = NormIdentity

        # (bsz, 224, 224, 1)
        x = conv(self.num_filters, (1, 1), (1, 1))(x)
        x = conv(self.num_filters, (3, 3), (1, 1))(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        # (bsz, 224, 224, 64)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        # (bsz, 112, 112, 64)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # strides=[1, 2, 2, ...] for conv3x3 in `Bottleneck`
                #     stride can be seen as downsample scale/rate
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                # conv2_ksize for conv3x3 in `Bottleneck`
                #     first `num_conv3x3` are conv3x3
                num_conv3x3 = self.num_conv3x3_per_stage[i]
                conv2_ksize = 3 if j < num_conv3x3 else 1

                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act,
                                   conv2_ksize=conv2_ksize)(x)

            # i=0 (bsz, 112, 112,  256)
            # i=1 (bsz,  56,  56,  512)
            # i=2 (bsz,  28,  28, 1024)
            # i=3 (bsz,  14,  14, 2048)
        x = np.mean(x, axis=(1, 2))
        # (bsz, 2048)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        # (bsz, num_classes)
        x = np.asarray(x, self.dtype)
        return x

    @property
    def receptive_field(self):
        filtered = list(filter(lambda x: x[0] == self.num_conv3x3_per_stage,
                               bagnet_num_conv3x3_and_receptive_fields))
        if len(filtered) != 1:
            raise ValueError(f'{self.num_conv3x3_per_stage} not supported')
        else:
            return filtered[0][1]


bagnet_num_conv3x3_and_receptive_fields = [
    ([1, 1, 0, 0], 11),
    ([1, 1, 1, 0], 19),
    ([1, 1, 1, 1], 35),
    ([2, 1, 1, 1], 39),
    ([2, 2, 1, 1], 47),
    ([2, 2, 2, 1], 63),
    ([2, 2, 2, 2], 95),
]


def _bagnet(resnet_version, receptive_field, trunk):
    if resnet_version == 'resnet18':
        stage_sizes = [2, 2, 2, 2]
        block_cls = BagNetBlock
    elif resnet_version == 'resnet50':
        stage_sizes = [3, 4, 6, 3]
        block_cls = BottleneckBagNetBlock

    filtered = list(filter(lambda x: x[1] == receptive_field,
                           bagnet_num_conv3x3_and_receptive_fields))
    if len(filtered) != 1:
        raise ValueError(f'receptive_field={receptive_field} not supported')
    else:
        num_conv3x3_per_stage = filtered[0][0]

    if trunk:
        backbone = BagNetTrunk
    else:
        backbone = BagNet

    return partial(backbone, stage_sizes=stage_sizes,
                   block_cls=block_cls,
                   num_conv3x3_per_stage=num_conv3x3_per_stage,)



BagNet18x11Trunk = _bagnet('resnet18', 11, True)
BagNet18x19Trunk = _bagnet('resnet18', 19, True)
BagNet18x35Trunk = _bagnet('resnet18', 35, True)
BagNet18x47Trunk = _bagnet('resnet18', 47, True)
BagNet18x63Trunk = _bagnet('resnet18', 63, True)
BagNet18x95Trunk = _bagnet('resnet18', 95, True)
BagNet50x11Trunk = _bagnet('resnet50', 11, True)
BagNet50x19Trunk = _bagnet('resnet50', 19, True)
BagNet50x35Trunk = _bagnet('resnet50', 35, True)
BagNet50x47Trunk = _bagnet('resnet50', 47, True)
BagNet50x63Trunk = _bagnet('resnet50', 63, True)
BagNet50x95Trunk = _bagnet('resnet50', 95, True)


BagNet18x11 = _bagnet('resnet18', 11, False)
BagNet18x19 = _bagnet('resnet18', 19, False)
BagNet18x35 = _bagnet('resnet18', 35, False)
BagNet18x47 = _bagnet('resnet18', 47, False)
BagNet18x63 = _bagnet('resnet18', 63, False)
BagNet18x95 = _bagnet('resnet18', 95, False)
BagNet50x11 = _bagnet('resnet50', 11, False)
BagNet50x19 = _bagnet('resnet50', 19, False)
BagNet50x35 = _bagnet('resnet50', 35, False)
BagNet50x47 = _bagnet('resnet50', 47, False)
BagNet50x63 = _bagnet('resnet50', 63, False)
BagNet50x95 = _bagnet('resnet50', 95, False)


def print_num_params():
    # Prints number of parameters for some models 
    #
    #     #parameters for models((224, 224, 3))
    #     ResNet18Trunk       : 11.18 M
    #     ResNet34Trunk       : 21.28 M
    #     ResNet50Trunk       : 23.51 M
    #
    #     BagNet18x11Trunk    :  1.54 M
    #     BagNet18x19Trunk    :  1.80 M
    #     BagNet18x35Trunk    :  2.85 M
    #     BagNet18x47Trunk    :  3.01 M
    #     BagNet18x63Trunk    :  3.54 M
    #     BagNet18x95Trunk    :  5.63 M
    #     BagNet50x11Trunk    : 13.64 M
    #     BagNet50x19Trunk    : 14.16 M
    #     BagNet50x35Trunk    : 16.26 M
    #     BagNet50x47Trunk    : 16.43 M
    #     BagNet50x63Trunk    : 16.95 M
    #     BagNet50x95Trunk    : 19.05 M
    #
    key = random.PRNGKey(0)
    resnet_trunks = [
        ('ResNet18Trunk', ResNet18Trunk),
        ('ResNet34Trunk', ResNet34Trunk),
        ('ResNet50Trunk', ResNet50Trunk),
    ]

    bagnet_trunks = [
        ('BagNet18x11Trunk', BagNet18x11Trunk),
        ('BagNet18x19Trunk', BagNet18x19Trunk),
        ('BagNet18x35Trunk', BagNet18x35Trunk),
        ('BagNet18x47Trunk', BagNet18x47Trunk),
        ('BagNet18x63Trunk', BagNet18x63Trunk),
        ('BagNet18x95Trunk', BagNet18x95Trunk),
        ('BagNet50x11Trunk', BagNet50x11Trunk),
        ('BagNet50x19Trunk', BagNet50x19Trunk),
        ('BagNet50x35Trunk', BagNet50x35Trunk),
        ('BagNet50x47Trunk', BagNet50x47Trunk),
        ('BagNet50x63Trunk', BagNet50x63Trunk),
        ('BagNet50x95Trunk', BagNet50x95Trunk),
    ]

    in_shape = (224, 224, 3)
    print(f'#parameters for models({in_shape})')
    print()
    for name, trunk_def in resnet_trunks+bagnet_trunks:
        m = trunk_def()
        params = m.init(key, random.normal(key, in_shape))
        num_params = pytree_num_parameters(params['params'])
        print(f'{name:20}: {num_params/1e6:5.2f} M')
        



## Regular Convolutional Networks 


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, kernel_size=(4, 4), strides=(2, 2))
        assert(x.shape[1] == 224 and x.shape[2] == 224)
        x = x.reshape(-1, 224, 224, 1)
        # (1, 224, 224, 1)
        x = conv(features=16)(x)
        x = nn.relu(x)
        x = conv(features=32)(x)
        x = nn.relu(x)
        x = conv(features=64)(x)
        x = nn.relu(x)
        x = conv(features=128)(x)
        x = nn.relu(x)
        # (1, 14, 14, 128)
        x = np.mean(x, axis=(1, 2))
        # (1, 128)
        return x


class CNNMnist(nn.Module):
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class CNNMnistTrunk(nn.Module):
    # in_shape: (1, 28, 28, 1)
    # receptive field: (10, 10)

    @nn.compact
    def __call__(self, x):
        # (N, H, W, C)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # (N, Px, Py, L)
        return x

    @property
    def receptive_field(self):
        return 10
