from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv_lib.utils.data_type import _SIZE_
from cv_lib.basebone.basic_blocks.basic_conv import build_basic_block

class OctConvMidBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=True, alpha_in:float=0.5, alpha_out:float=0.5,
     activations:Optional[nn.Module]=nn.ReLU(True)):
        super(OctConvMidBlock, self).__init__()

        h_in_channels: int = int((1 - alpha_in) * in_channels)
        l_in_channels: int = int(alpha_in * in_channels)

        h_out_channels: int = int((1 - alpha_out) * out_channels)
        l_out_channels: int = int(alpha_out * out_channels)

        self.h2h_conv: nn.Module = build_basic_block(h_in_channels, h_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        self.h2l_conv: nn.Module = build_basic_block(h_in_channels, l_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        self.l2h_conv: nn.Module = build_basic_block(l_in_channels, h_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        self.l2l_conv: nn.Module = build_basic_block(l_in_channels, l_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)

        self.activations: Optional[nn.Module] = activations

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:

        h_x, l_x = x

        h2h_output: torch.Tensor = self.h2h_conv(h_x)
        h2l_output: torch.Tensor = F.avg_pool2d(self.h2l_conv(h_x), 2)

        h_size = h2h_output.size()[-2:]

        l2h_output: torch.Tensor = F.interpolate(self.l2h_conv(l_x), size=h_size)
        l2l_output: torch.Tensor = self.l2l_conv(l_x)

        h_output: torch.Tensor = h2h_output + l2h_output
        l_output: torch.Tensor = h2l_output + l2l_output

        if self.activations:
            h_output: torch.Tensor = self.activations(h_output)
            l_output: torch.Tensor = self.activations(l_output)


        return h_output, l_output

class OctConvInBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=0, dilation:_SIZE_=1, groups:int=1, bias:bool=True, alpha_out:float=0.5,
     activations:Optional[nn.Module]=nn.ReLU(True)):
        super(OctConvInBlock, self).__init__()

        h_out_channels = int((1 - alpha_out) * out_channels)
        l_out_channels = int(alpha_out * out_channels)

        self.h2h_conv: nn.Module = build_basic_block(in_channels, h_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        self.h2l_conv: nn.Module = build_basic_block(in_channels, l_out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        
        self.activations: Optional[nn.Module] = activations

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        h_output: torch.Tensor = self.h2h_conv(x)
        l_output: torch.Tensor = F.avg_pool2d(self.h2l_conv(x), 2)

        if self.activations:
            h_output: torch.Tensor = self.activations(h_output)
            l_output: torch.Tensor = self.activations(l_output)

        return h_output, l_output

class OctConvOutBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=0, dilation:_SIZE_=1, groups:int=1, bias:bool=True, alpha_in:float=0.5,
     activations:Optional[nn.Module]=nn.ReLU(True)):
         
        super(OctConvOutBlock, self).__init__()

        h_in_channels = int((1 - alpha_in) * in_channels)
        l_in_channels = int(alpha_in * in_channels)

        self.h2h_conv: nn.Module = build_basic_block(h_in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        self.l2h_conv: nn.Module = build_basic_block(l_in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, None)
        
        self.activations: Optional[nn.Module] = activations

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        h_x, l_x = x

        h2h_output: torch.Tensor = self.h2h_conv(h_x)
        h_size = h2h_output.size()[-2:]

        l2h_output: torch.Tensor = F.interpolate(self.l2h_conv(l_x), size=h_size)

        h_output = h2h_output + l2h_output

        if self.activations:
            h_output: torch.Tensor = self.activations(h_output)

        return h_output

def build_oct_block(in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=0, dilation:_SIZE_=1, groups:int=1, bias:bool=True, alpha_in:float=0.5, alpha_out:float=0.5,
     activations:Optional[nn.Module]=nn.ReLU(True)):
    #assert (alpha_in == 0 and alpha_out == 0)
    if alpha_in == 0:
        return OctConvInBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, alpha_out, activations)
    elif alpha_out == 0:
        return OctConvOutBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, alpha_in, activations)
    else:
        return OctConvMidBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, alpha_in, alpha_out, activations)
