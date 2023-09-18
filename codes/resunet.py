import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

import numpy as np

from collections import OrderedDict
from functools import partial


# ResNet encoder implementation adopted from https://github.com/FrancescoSaverioZuppichini/ResNet.
# Decoder implementation adopted from smp (https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders/unet).


class Conv2dAuto(nn.Conv2d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)


def conv_bn(inplanes, planes, kernel_size=3, stride=1, *args, **kwargs):
    conv = partial(Conv2dAuto, kernel_size=kernel_size, stride=stride, bias=False)
    return nn.Sequential(OrderedDict(
        {
            'conv' : conv(inplanes, planes, *args, **kwargs),
            'bn' : nn.BatchNorm2d(planes)
        }
    ))


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__()
        self.inplanes, self.planes = inplanes, planes
        self.expansion, self.downsampling = expansion, downsampling

        self.blocks = nn.Identity()
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv_bn' : conv_bn(self.inplanes, self.expansion * self.planes, kernel_size=1, stride=self.downsampling)
            }
        )) if self.inplanes != self.expansion * self.planes else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x



class ResNetBasicBlock(ResNetBlock):
    def __init__(self, inplanes, planes, kernel_size=3, activation=nn.ReLU, *args, **kwargs):
        super().__init__(inplanes, planes, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.inplanes, self.planes, kernel_size=kernel_size, stride=self.downsampling),
            activation(),
            conv_bn(self.planes, self.expansion * self.planes, kernel_size=kernel_size)
        )



class ResNetBottleneckBlock(ResNetBlock):
    expansion = 4
    def __init__(self, inplanes, planes, kernel_size=3, activation=nn.ReLU, *args, **kwargs):
        super().__init__(inplanes, planes, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.inplanes, self.planes, kernel_size=1),
            activation(),
            conv_bn(self.planes, self.planes, kernel_size=kernel_size, stride=self.downsampling),
            activation(),
            conv_bn(self.planes, self.expansion * self.planes, kernel_size=1)
        )



class ResNetLayer(nn.Module):
    def __init__(self, inplanes, planes, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        self.downsampling = 2 if inplanes != planes else 1

        self.blocks = nn.Sequential(
            block(inplanes, planes, *args, **kwargs, downsampling=self.downsampling),
            *[block(block.expansion * planes, block.expansion * planes, downsamplig=1, *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x



class ResNetEncoder(nn.Module):
    def __init__(self, inplanes=3, stem_size=64, block_sizes=[64,128,256,512], depths=[2,2,2,2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.out_channels = [stem_size] + block_sizes if stem_size else block_sizes

        # no stem if stem_size=0
        if stem_size:
            self.stem = nn.Sequential(
                conv_bn(inplanes, stem_size, kernel_size=7, stride=2),
                activation(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            inplanes = stem_size

        else:
            self.stem = nn.Identity()

        in_out_planes = list(zip(block_sizes[:-1], block_sizes[1:]))
        self.layers = nn.ModuleList([
            ResNetLayer(inplanes, block_sizes[0], n=depths[0], activation=activation, block=block, *args, **kwargs),
            *[ResNetLayer(inplanes*block.expansion, planes, n=n, activation=activation, block=block, *args, **kwargs)
            for (inplanes, planes), n in zip(in_out_planes, depths[1:])]
        ])

        downsampling = [layer.downsampling for layer in self.layers]
        self.downsampling = [4] + downsampling if stem_size else downsampling


    def forward(self, x):
        x = self.stem(x)
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs = [x] + outputs
        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, depth=1, scale_mode='nearest'):
        super().__init__()
        self.in_channel, self.skip_channel, self.out_channel = in_channel, skip_channel, out_channel

        blocks = [conv_bn(in_channel + skip_channel, out_channel)] + [conv_bn(out_channel, out_channel)]*depth
        self.blocks = nn.Sequential(*blocks)
        self.scale_mode = 'nearest'


    def forward(self, x, skip=None, scale_factor=2):
        if self.skip_channel:
            x = F.interpolate(x, scale_factor=scale_factor, mode=self.scale_mode)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=scale_factor, mode=self.scale_mode)
        x = self.blocks(x)
        return x



class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, depth):
        super().__init__()

        skip_channels = list(encoder_channels[1:]+[0])
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        blocks = [
            DecoderBlock(in_channel, skip_channel, out_channel, depth)
            for in_channel, skip_channel, out_channel in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, *features):
        x = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            if i < len(skips):
                x = decoder_block(x, skip=skips[i])
            else:
                x = decoder_block(x, skip=None)
        return x



class UNet(nn.Module):
    def __init__(self,
                 encoder=ResNetEncoder(),
                 decoder_channels=[256,128,64,32,16],
                 activation=nn.Sigmoid,
                 postprocessing=nn.Identity,
                 decoder_depth=1,
                 aux=False):
        super().__init__()

        self.encoder = encoder
        # For simplification, just assume the decoder reduces the channels by two at all layers,
        # with the last layer outputting 16 channels.
        encoder_channels = encoder.out_channels[::-1]
        self.decoder = UNetDecoder(encoder_channels, decoder_channels, decoder_depth)
        self.segmentation_head = nn.Sequential(OrderedDict(
            {
                'conv_bn':  conv_bn(decoder_channels[-1], 1),
                'activation': activation(),
                'post-processing': postprocessing()
            }
        ))
        if aux:
            self.classification_head = nn.Sequential(
                                            pool = nn.AdaptiveAvgPool2d(1),
                                            flatten = nn.Flatten(),
                                            dropout = nn.Dropout(p=0.2, inplace=True),
                                            linear = nn.Linear(encoder.out_channels[-1], 1, bias=True),
                                            activation = nn.Sigmoid()
                                            )
        else:
            self.classification_head = None


    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features[::-1])
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks


    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x