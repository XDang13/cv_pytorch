from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.utils.data_type import _SIZE_, _CONFIG_
from cv_lib.basebone.basic_blocks.basic_conv import build_basic_block
from cv_lib.basebone.basic_blocks.oct_conv import build_oct_block
from cv_lib.basebone.basic_blocks.deform_conv import build_deform_conv_block
from cv_lib.utils.configs.darknet import darknet_53_basic_conv

class DarkNetResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DarkNetResBlock, self).__init__()
        mid_channels = in_channels // 2
        self.block_1 = build_basic_block(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, activations=nn.LeakyReLU())
        self.block_2 = build_basic_block(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, activations=nn.LeakyReLU())

    def forward(self, tensor: torch.Tensor):
        output = self.block_1(tensor)
        output = self.block_2(output)

        return output + tensor

class DarkNetBaseBone(nn.Module):
    def build_basic_block_layer(self, config: _CONFIG_):
        channels: List[int] = config['channels']
        kernel_sizes: List[_SIZE_] = config['kernel_sizes']
        strides: List[_SIZE_] = config['stride']
        paddings: List[_SIZE_] = config['paddings']
        dilations: List[int] = config['dilations']
        groups: List[int] = config['groups']
        biases: List[bool] = config['biases']

        layers: List[nn.Module] = []

        for channel, kernel_size, stride, padding, dilation, group, bias in zip(
            channels, kernel_sizes, strides, paddings, dilations, groups, biases
        ):
            layers.append(build_basic_block(
                self.channel, channel, kernel_size, stride, padding,
                dilation, group, bias, self.activations
            ))
            self.channel = channel

        return nn.Sequential(*layers)

    def build_res_block_layer(self, config: _CONFIG_):
        channels: List[int] = config['channels']
        kernel_sizes: List[_SIZE_] = config['kernel_sizes']
        strides: List[_SIZE_] = config['stride']
        paddings: List[_SIZE_] = config['paddings']
        dilations: List[int] = config['dilations']
        groups: List[int] = config['groups']
        biases: List[bool] = config['biases']

        layers: List[nn.Module] = [build_basic_block(self.channel, channels[0],
         kernel_sizes[0], strides[0], paddings[0], dilations[0], groups[0], biases[0], self.activations)]
        self.channel = channels[0]

        for channel in channels[1:]:
            layers.append(DarkNetResBlock(self.channel, channel))
            self.channel = channel

        return nn.Sequential(*layers)

class DarkNet53(DarkNetBaseBone):
    def __init__(self, in_channels:int, config:_CONFIG_, activations: Optional[nn.Module]=nn.LeakyReLU()):
        super(DarkNet53, self).__init__()
        self.channel = in_channels
        self.activations = activations

        self.block_1 = build_basic_block(self.channel, 32, 3, 1, 1, activations=self.activations)
        self.channel = 32
        self.block_2 = self.build_res_block_layer(config[0])
        self.block_3 = self.build_res_block_layer(config[1])
        self.block_4 = self.build_res_block_layer(config[2])
        self.block_5 = self.build_res_block_layer(config[3])
        self.block_6 = self.build_res_block_layer(config[4])

    def forward(self, tensor: torch.Tensor):
        output = self.block_1(tensor)
        #output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_2(output)
        #output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_3(output)
        #output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_4(output)
        #output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_5(output)
        #output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_6(output)

        return output

    def to_list(self) -> List[nn.Sequential]:
        return [self.block_1, self.block_2, self.block_3,
                self.block_4, self.block_5, self.block_6]

def build_darknet_53_basic_conv(channel: int, activations: Optional[nn.Module]=nn.LeakyReLU()):
    return DarkNet53(channel, darknet_53_basic_conv, activations)