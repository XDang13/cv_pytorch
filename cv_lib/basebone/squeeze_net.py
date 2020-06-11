from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.utils.data_type import _SIZE_, _CONFIG_
from cv_lib.basebone.basic_blocks.basic_conv import build_basic_block
from cv_lib.basebone.basic_blocks.oct_conv import build_oct_block
from cv_lib.basebone.basic_blocks.deform_conv import build_deform_conv_block

class SqueezeBlockWBasicConv(nn.Module):
    def __init__(self, in_channels: int, channels: int, expand1x1_channels: int, expand3x3_channels: int, activations: Optional[nn.Module]=nn.ReLU(True)):
        super(SqueezeBlockWBasicConv, self).__init__()

        self.sequeeze_block = build_basic_block(in_channels, channels, kernel_size=1,
         padding=0, bias=True, activations=activations)

        self.expand1x1_block = build_basic_block(channels, expand3x3_channels, kernel_size=1,
         padding=0, bias=True, activations=activations)

        self.expand3x3_block = build_basic_block(channels, expand3x3_channels, kernel_size=3,
         padding=1, bias=True, activations=activations)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        squeeze_tensor = self.sequeeze_block(tensor)
        expand1x1_tensor = self.expand1x1_block(squeeze_tensor)
        expand3x3_tensor = self.expand3x3_block(squeeze_tensor)

        output = torch.cat([expand1x1_tensor, expand3x3_tensor], dim=1)

        return output

class SqueezeBlockWOctConv(nn.Module):
    def __init__(self, in_channels: int, channels: int, expand1x1_channels: int, expand3x3_channels: int, alpha: float, activations: Optional[nn.Module]=nn.ReLU(True)):
        super(SqueezeBlockWOctConv, self).__init__()

        self.sequeeze_block = build_oct_block(in_channels, channels, kernel_size=1, padding=0,
         bias=True, alpha_in=0, alpha_out=alpha, activations=activations)

        self.expand1x1_block = build_oct_block(channels, expand1x1_channels, kernel_size=1, padding=0,
         bias=True, alpha_in=alpha, alpha_out=0, activations=activations)

        self.expand3x3_block = build_oct_block(channels, expand3x3_channels, kernel_size=3, padding=1,
         bias=True, alpha_in=alpha, alpha_out=0, activations=activations)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        squeeze_tensor = self.sequeeze_block(tensor)
        expand1x1_tensor = self.expand1x1_block(squeeze_tensor)
        expand3x3_tensor = self.expand3x3_block(squeeze_tensor)

        output = torch.cat([expand1x1_tensor, expand3x3_tensor], dim=1)

        return output

class SqueezeBlockWDeformConv(nn.Module):
    def __init__(self, in_channels: int, channels: int, expand1x1_channels: int, expand3x3_channels: int, activations: Optional[nn.Module]=nn.ReLU(True)):
        super(SqueezeBlockWDeformConv, self).__init__()

        self.sequeeze_block = build_deform_conv_block(in_channels, channels, kernel_size=1,
         padding=0, bias=True, activations=activations)

        self.expand1x1_block = build_deform_conv_block(channels, expand3x3_channels, kernel_size=1,
         padding=0, bias=True, activations=activations)

        self.expand3x3_block = build_deform_conv_block(channels, expand3x3_channels, kernel_size=3,
         padding=1, bias=True, activations=activations)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        squeeze_tensor = self.sequeeze_block(tensor)
        expand1x1_tensor = self.expand1x1_block(squeeze_tensor)
        expand3x3_tensor = self.expand3x3_block(squeeze_tensor)

        output = torch.cat([expand1x1_tensor, expand3x3_tensor], dim=1)

        return output

class SqueezeNetBasebone(nn.Module):
    def make_basic_block_layer(self, config: _CONFIG_) -> nn.Sequential:

        channels: List[int] = config['channels']

        layers: List[nn.Module] = []

        for channel, kernel_size, stride, padding, dilation, group, bias in zip(
            channels, kernel_sizes, strides, paddings, dilations, groups, biases
        ):
            layers.append(
                
            )
            self.channel = channel

        return nn.Sequential(*layers)

    def make_oct_block_layer(self, config: _CONFIG_) -> nn.Sequential:
        pass
    def make_deform_block_layer(self, config: _CONFIG_) -> nn.Sequential:
        pass

    def make_layer(self, config: _CONFIG_) -> nn.Sequential:
        pass