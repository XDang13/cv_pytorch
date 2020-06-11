from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOV3Criterion(nn.Module):
    def __init__(self, no_obj_weight: float=0.5, xy_weight:float=2.5, wh_weight:float=2.5, confidence_weight:float=1, cla_weight:float=1):
        super(YOLOV3Criterion, self).__init__()
        self.no_obj_weight = no_obj_weight
        self.xy_weight = xy_weight
        self.wh_weight = wh_weight
        self.confidence_weight = confidence_weight
        self.cla_weight = cla_weight
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, obj_masks: torch.Tensor, no_obj_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_preds = preds[..., 0:1]
        y_preds = preds[..., 1:2]
        w_preds = preds[..., 2:3]
        h_preds = preds[..., 3:4]
        obj_preds = preds[..., 4:5]
        cla_preds = preds[..., 5:]
        
        x_targets = targets[..., 0:1]
        y_targets = targets[..., 1:2]
        w_targets = targets[..., 2:3]
        h_targets = targets[..., 3:4]
        obj_targets = targets[..., 4:5]
        cla_targets = targets[..., 5:]
        
        x_loss = F.binary_cross_entropy_with_logits(x_preds[obj_masks], x_targets[obj_masks])
        y_loss = F.binary_cross_entropy_with_logits(y_preds[obj_masks], y_targets[obj_masks])
        w_loss = F.mse_loss(w_preds[obj_masks], w_targets[obj_masks])
        h_loss = F.mse_loss(h_preds[obj_masks], h_targets[obj_masks])
        obj_loss = F.binary_cross_entropy_with_logits(obj_preds[obj_masks], obj_targets[obj_masks])
        no_obj_loss = F.binary_cross_entropy_with_logits(obj_preds[no_obj_masks], obj_targets[no_obj_masks])
        confidence_loss = obj_loss + no_obj_loss * self.no_obj_weight
        cla_loss = F.binary_cross_entropy_with_logits(cla_preds[obj_masks], cla_targets[obj_masks])
        loss = (x_loss * self.xy_weight + y_loss * self.xy_weight + w_loss * self.wh_weight + h_loss * self.wh_weight + confidence_loss * self.confidence_weight + cla_loss * self.cla_weight)
        

        return loss, (x_loss.item(), y_loss.item(), w_loss.item(), h_loss.item(), obj_loss.item(), no_obj_loss.item(), cla_loss.item())
