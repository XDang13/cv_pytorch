from typing import List
from cv_lib.utils.data_type import _GRAD_

vggbasebone_16_basic_conv: List[_GRAD_] = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'basic_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'basic_block'
    },
    {
        'channels': [256, 256, 256],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'basic_block'
    }
]

vggbasebone_19_basic_conv: List[_GRAD_] = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'basic_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'basic_block'
    },
    {
        'channels': [256, 256, 256, 256],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'basic_block'
    },
]

vggbasebone_16_oct_conv = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'alpha_ins': [0, 0.25],
        'alpha_outs': [0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'alpha_ins': [0, 0.25],
        'alpha_outs': [0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [256, 256, 256],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'alpha_ins': [0, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'alpha_ins': [0, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'alpha_ins': [0, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0],
        'type': 'oct_block'
    },
]

vggbasebone_19_oct_conv = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'alpha_ins': [0, 0.25],
        'alpha_outs': [0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'alpha_ins': [0, 0.25],
        'alpha_outs': [0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [256, 256, 256, 256],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'alpha_ins': [0, 0.25, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'alpha_ins': [0, 0.25, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0.25, 0],
        'type': 'oct_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'alpha_ins': [0, 0.25, 0.25, 0.25],
        'alpha_outs': [0.25, 0.25, 0.25, 0],
        'type': 'oct_block'
    },
]

vggbasebone_16_deform_conv: List[_GRAD_] = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'deform_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'deform_block'
    },
    {
        'channels': [256, 256, 256],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'deform_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'deform_block'
    },
    {
        'channels': [512, 512, 512],
        'kernel_sizes': [3, 3, 3],
        'stride': [1, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'deform_block'
    }
]

vggbasebone_19_deform_conv: List[_GRAD_] = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'deform_block'
    },
    {
        'channels': [128, 128],
        'kernel_sizes': [3, 3],
        'stride': [1, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'deform_block'
    },
    {
        'channels': [256, 256, 256, 256],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'deform_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'deform_block'
    },
    {
        'channels': [512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'stride': [1, 1, 1, 1],
        'paddings': [1, 1, 1, 1],
        'dilations': [1, 1, 1, 1],
        'groups': [1, 1, 1, 1],
        'biases': [False, False, False, False],
        'type': 'deform_block'
    },
]