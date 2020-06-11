from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.utils.data_type import _SIZE_, _CONFIG_
from cv_lib.basebone.basic_blocks.basic_conv import build_basic_block
from cv_lib.basebone.basic_blocks.oct_conv import build_oct_block
from cv_lib.basebone.basic_blocks.deform_conv import build_deform_conv_block
from cv_lib.utils.configs.vgg import *

class VGGBasebone(nn.Module):
    def build_block(self, config: _CONFIG_) -> nn.Sequential:
        if config['type'] == 'basic_block':
            block = self.make_baisc_conv_block_layer(config)
        elif config['type'] == 'oct_block':
            block = self.make_oct_conv_block_layer(config)
        elif config['type'] == 'deform_block':
            block = self.make_deform_conv_block_layer(config)

        return block
    
    def make_baisc_conv_block_layer(self, config: _CONFIG_) -> nn.Sequential:
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

    def make_oct_conv_block_layer(self, config: _CONFIG_) -> nn.Sequential:
        channels: List[int] = config['channels']
        kernel_sizes: List[_SIZE_] = config['kernel_sizes']
        strides: List[_SIZE_] = config['stride']
        paddings: List[_SIZE_] = config['paddings']
        dilations: List[int] = config['dilations']
        groups: List[int] = config['groups']
        biases: List[bool] = config['biases']
        alpha_ins: List[float] = config['alpha_ins']
        alpha_outs: List[float] = config['alpha_outs']

        layers: List[nn.Module] = []

        for channel, kernel_size, stride, padding, dilation, group, bias, alpha_in, alpha_out in zip(
            channels, kernel_sizes, strides, paddings, dilations, groups, biases, alpha_ins, alpha_outs
        ):
            layers.append(
                build_oct_block(
                    self.channel, channel, kernel_size, stride, padding, dilation,
                    group, bias, alpha_in, alpha_out, self.activations
                )
            )

            self.channel = channel

        return nn.Sequential(*layers)

    def make_deform_conv_block_layer(self, config: _CONFIG_) -> nn.Sequential:
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
            layers.append(
                build_deform_conv_block(
                    self.channel, channel, kernel_size, stride, padding,
                    dilation, group, bias, self.activations
                )
            )

            self.channel = channel

        return nn.Sequential(*layers)


class DefaultVGGBasebone(VGGBasebone):
    def __init__(self, channel: int, configs, activations: Optional[nn.Module]=nn.ReLU(True)):
        super(DefaultVGGBasebone, self).__init__()
        self.channel = channel
        self.activations: Optional[nn.Module] = activations
        self.block_1 = self.build_block(configs[0])
        self.block_2 = self.build_block(configs[1])
        self.block_3 = self.build_block(configs[2])
        self.block_4 = self.build_block(configs[3])
        self.block_5 = self.build_block(configs[4])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.block_1(tensor)
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_2(output)
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_3(output)
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_4(output)
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = self.block_5(output)
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        
        return output

    def to_list(self) -> List[nn.Sequential]:
        return [self.block_1, self.block_2, self.block_3,
                self.block_4, self.block_5]
    
    def to_sequential(self) -> nn.Sequential:
        
        return nn.Sequential(*self.to_list())
    
    def to_modellist(self) -> nn.ModuleList:
        
        return nn.ModuleList(self.to_list())

def build_vggbasebone_16_basic_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_16_basic_conv, activations)

def build_vggbasebone_19_basic_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_19_basic_conv, activations)

def build_vggbasebone_16_oct_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_16_oct_conv, activations)

def build_vggbasebone_19_oct_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_19_oct_conv, activations)

def build_vggbasebone_16_deform_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_16_deform_conv, activations)

def build_vggbasebone_19_deform_conv(channel: int, activations: Optional[nn.Module]=nn.ReLU(True)):
    return DefaultVGGBasebone(channel, vggbasebone_19_deform_conv, activations)