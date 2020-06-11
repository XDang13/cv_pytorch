from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv_lib.utils.data_type import _SIZE_, _GRAD_

class DeformConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=False):
        super(DeformConv2D, self).__init__()
        assert(dilation == 1)
        assert(groups == 1)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.zero_padding: nn.ZeroPad2d = nn.ZeroPad2d(padding)
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=kernel_size, bias=bias)
        if type(kernel_size) is tuple:
            shift_kernel: int = kernel_size[0] * kernel_size[1]
        else:
            shift_kernel: int = kernel_size * kernel_size
        self.shift_conv: nn.Conv2d = nn.Conv2d(in_channels, 2 * shift_kernel, 3, stride=stride, padding=1)
        nn.init.constant_(self.shift_conv.weight, 0)
        self.shift_conv.register_backward_hook(self._set_lr)


        self.modulation_conv: nn.Conv2d = nn.Conv2d(in_channels, shift_kernel, 3, stride=stride, padding=1)
        nn.init.constant_(self.modulation_conv.weight, 0)
        self.modulation_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module: nn.Module, grad_input: _GRAD_, grad_output: _GRAD_):
        grad_input = (item * 0.1 for item in grad_input)
        grad_output = (item * 0.1 for item in grad_output)


    def get_position(self, offset: torch.Tensor, dtype: Union[str, torch.Tensor]) -> torch.Tensor:
        channels, height, width = offset.size(1) // 2, offset.size(2), offset.size(3)

        position_n = self.get_position_n(channels, dtype)
        position_0 = self.get_position_0(height, width, channels, dtype)

        position = position_0 + position_n + offset
        

        return position


    def get_position_n(self, channels: int, dtype: Union[str, torch.Tensor]) -> torch.Tensor:
        if type(self.kernel_size) is tuple:
            point_x: int = (self.kernel_size[0] - 1) // 2
            point_y: int = (self.kernel_size[1] - 1) // 2
        else:
            point_x: int = (self.kernel_size - 1) // 2
            point_y: int = (self.kernel_size - 1) // 2
        position_n_x, position_n_y = torch.meshgrid(
            torch.arange(-point_x, point_x + 1),
            torch.arange(-point_y, point_y + 1)
        )

        position_n = torch.cat([torch.flatten(position_n_x), torch.flatten(position_n_y)], 0)
        position_n = position_n.view(1, 2*channels, 1, 1).type(dtype)

        return position_n

    def get_position_0(self, height: int, width: int, channels: int, dtype: Union[str, torch.Tensor]) -> torch.Tensor:
        if type(self.stride) is tuple:
            stride_x: int = self.stride[0]
            stride_y: int = self.stride[1]
        else:
            stride_x: int = self.stride
            stride_y: int = self.stride

        position_0_x, position_0_y = torch.meshgrid(
            torch.arange(1, height * stride_x + 1, stride_x),
            torch.arange(1, width * stride_y + 1, stride_y)
        )

        position_0_x = torch.flatten(position_0_x).view(1, 1, height, width).repeat(1, channels, 1, 1)
        position_0_y = torch.flatten(position_0_y).view(1, 1, height, width).repeat(1, channels, 1, 1)

        position_0 = torch.cat([position_0_x, position_0_y], 1).type(dtype)

        return position_0

    def get_x_q(self, x: torch.Tensor, q: torch.Tensor, channels: int) -> torch.Tensor:
        batch, height, width, _ = q.size()
        padded_width = x.size(3)
        x_channels = x.size(1)

        x = x.contiguous().view(batch, x_channels, -1)

        index = q[..., :channels] * padded_width + q[..., channels:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, x_channels, -1, -1, -1).contiguous().view(batch, x_channels, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(batch, x_channels, height, width, channels)

        return x_offset

    @staticmethod
    def reshape_x_offset(x_offset, kernel_size: _SIZE_):
        
        '''
        if type(kernel_size) is tuple:
            kernel_size_x: int = kernel_size[0]
            kernel_size_y: int = kernel_size[1]
        else:
            kernel_size_x: int = kernel_size
            kernel_size_y: int = kernel_size
        '''

        batch, channels, height, width, N = x_offset.size()

        x_offset = torch.cat(
            [x_offset[..., s: s+kernel_size].contiguous().view(batch, channels, height, width * kernel_size) for s in range(0, N, kernel_size)],
            dim=-1
        )

        x_offset = x_offset.contiguous().view(batch, channels, height * kernel_size, width * kernel_size)

        return x_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.shift_conv(x)
        modulation = torch.sigmoid(self.modulation_conv(x))

        dtype = offset.data.type()
        kernel_size = self.kernel_size
        _, channels, height, width = offset.size()
        channels = channels // 2

        if self.padding:
            x = self.zero_padding(x)

        position = self.get_position(offset, dtype)

        position = position.contiguous().permute(0, 2, 3, 1)
        left_top = position.detach().floor()
        right_bottom = left_top + 1

        left_top = torch.cat(
            [torch.clamp(left_top[..., :channels], 0, height-1),
             torch.clamp(left_top[..., channels:], 0, width-1)], dim=-1
        ).long()
        right_bottom = torch.cat(
            [torch.clamp(right_bottom[..., :channels], 0, height - 1),
             torch.clamp(right_bottom[..., channels:], 0, width - 1)], dim=-1
        ).long()

        left_bottom = torch.cat(
            [left_top[..., :channels], right_bottom[..., channels:]], dim=-1
        ).long()
        right_top = torch.cat(
            [right_bottom[..., :channels], left_top[..., channels:]], dim=-1
        ).long()

        position = torch.cat(
            [torch.clamp(position[..., :channels], 0, height - 1),
             torch.clamp(position[..., channels:], 0, width - 1)], dim=-1
        ).long()

        g_left_top = (
            (1 + (left_top[..., :channels].type_as(position) - position[..., :channels])) *
            (1 + (left_top[..., channels:].type_as(position) - position[..., channels:]))
        )
        g_right_bottom = (
            (1 - (right_bottom[..., :channels].type_as(position) - position[..., :channels])) *
            (1 - (right_bottom[..., channels:].type_as(position) - position[..., channels:]))
        )
        g_left_bottom = (
            (1 + (left_bottom[..., :channels].type_as(position) - position[..., :channels])) *
            (1 + (left_bottom[..., channels:].type_as(position) - position[..., channels:]))
        )
        g_right_top = (
            (1 - (right_top[..., :channels].type_as(position) - position[..., :channels])) *
            (1 - (right_top[..., channels:].type_as(position) - position[..., channels:]))
        )

        x_q_lt = self.get_x_q(x, left_top, channels)
        x_q_rb = self.get_x_q(x, right_bottom, channels)
        x_q_lb = self.get_x_q(x, left_bottom, channels)
        x_q_rt = self.get_x_q(x, right_top, channels)

        x_offset = (
            g_left_top.unsqueeze(dim=1) * x_q_lt +
            g_right_bottom.unsqueeze(dim=1) * x_q_rb +
            g_left_bottom.unsqueeze(dim=1) * x_q_lb +
            g_right_top.unsqueeze(dim=1) * x_q_rt
        )

        modulation = modulation.contiguous().permute(0, 2, 3, 1)
        modulation = modulation.unsqueeze(dim=1)
        modulation = torch.cat([modulation for _ in range(x_offset.size(1))], dim=1)

        x_offset *= modulation

        x_offset = self.reshape_x_offset(x_offset, kernel_size)
        out = self.conv(x_offset)

        return out

class DeformConv2dBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=False, activations:Optional[nn.Module]=nn.ReLU(True)):
        super(DeformConv2dBlock, self).__init__()

        self.deform_conv = DeformConv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.bn: Optional[nn.Module] = nn.BatchNorm2d(out_channels) if not bias else None

        self.activations = activations

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.deform_conv(tensor)
        if self.bn:
            output = self.bn(output)

        if self.activations:
            output = self.activations(output)

        return output


def build_deform_conv_block(in_channels:int, out_channels:int, kernel_size:_SIZE_, stride:_SIZE_=1,
     padding:_SIZE_=1, dilation:_SIZE_=1, groups:int=1, bias:bool=False,
     activations:Optional[nn.Module]=nn.ReLU(True)) -> DeformConv2dBlock:
    
    block: DeformConv2dBlock = DeformConv2dBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activations)

    return block