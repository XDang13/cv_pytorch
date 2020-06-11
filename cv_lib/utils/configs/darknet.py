from typing import List
from cv_lib.utils.data_type import _GRAD_

darknet_53_basic_conv: List[_GRAD_] = [
    {
        'channels': [64, 64],
        'kernel_sizes': [3, 3],
        'stride': [2, 1],
        'paddings': [1, 1],
        'dilations': [1, 1],
        'groups': [1, 1],
        'biases': [False, False],
        'type': 'basic_block'
    },
    {
        'channels': [128, 128, 128],
        'kernel_sizes': [3, 3, 3],
        'stride': [2, 1, 1],
        'paddings': [1, 1, 1],
        'dilations': [1, 1, 1],
        'groups': [1, 1, 1],
        'biases': [False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [256, 256, 256, 256, 256, 256, 256, 256, 256],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'stride': [2, 1, 1, 1, 1, 1, 1, 1, 1],
        'paddings': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'dilations': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'groups': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'biases': [False, False, False, False, False, False, False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [512, 512, 512, 512, 512, 512, 512, 512, 512],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'stride': [2, 1, 1, 1, 1, 1, 1, 1, 1],
        'paddings': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'dilations': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'groups': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'biases': [False, False, False, False, False, False, False, False, False],
        'type': 'basic_block'
    },
    {
        'channels': [1024, 1024, 1024, 1024, 1024],
        'kernel_sizes': [3, 3, 3, 3, 3],
        'stride': [2, 1, 1, 1, 1],
        'paddings': [1, 1, 1, 1, 1],
        'dilations': [1, 1, 1, 1, 1],
        'groups': [1, 1, 1, 1, 1],
        'biases': [False, False, False, False, False],
        'type': 'basic_block'
    }
]
