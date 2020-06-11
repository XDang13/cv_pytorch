from typing import Optional
import torch
import torch.nn as nn
import torch.functional as F
from cv_lib.utils.data_type import _SIZE_

class BasicConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=False, activations:Optional[nn.Module]=nn.ReLU(True)):
        super(BasicConv2D, self).__init__()
        self.conv2d: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn: Optional[nn.Module] = nn.BatchNorm2d(out_channels) if not bias else None

        self.activations = activations

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.conv2d(tensor)
        if self.bn:
            output = self.bn(output)

        if self.activations:
            output = self.activations(output)

        return output

def build_basic_block(in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=False,
     activations:Optional[nn.Module]=nn.ReLU(True)) -> BasicConv2D:

     block: BasicConv2D = BasicConv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activations)

     return block
