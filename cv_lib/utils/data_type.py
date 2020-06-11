from typing import Dict, List, Tuple, Union, Optional, TypeVar
import torch

_SIZE_ = TypeVar("_SIZE_", bound=Union[int, Tuple[int, int]])
_GRAD_ = TypeVar("_GRAD_", bound=Tuple[Optional[torch.Tensor]])
_CONFIG_ = TypeVar("_CONFIG_", bound=Dict["str", Union[str, List[Union[int, float, bool]]]])
_TENSOR_SIZE_ = TypeVar("_TENSOR_SIZE_", bound=List[Tuple[int, int]])
_IMG_SIZE_ = TypeVar("_IMG_SIZE_", bound=Tuple[int, int])
_ANCHOR_SIZE_ = TypeVar("_ANCHOR_SIZE_", bound=List[List[Tuple[int, int]]])