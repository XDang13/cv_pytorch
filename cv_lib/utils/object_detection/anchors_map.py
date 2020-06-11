from typing import List, Tuple

import numpy as np
from cv_lib.utils.data_type import _TENSOR_SIZE_, _IMG_SIZE_, _ANCHOR_SIZE_

class AnchorMap:
    def __init__(self, tensor_sizes: _TENSOR_SIZE_, img_size: _IMG_SIZE_, anchor_sizes: _ANCHOR_SIZE_, num_cla: int, threshold: float=0.5):
        self.num_maps = len(tensor_sizes)
        self.tensor_sizes = tensor_sizes
        self.img_size = img_size
        
        self.strides = self._build_strides()
        self.anchor_sizes = anchor_sizes
        self.anchors = self._build_anchor_maps()
        self.left_top_maps = self._build_left_top_maps()
        
        self.channels = num_cla + 5
        self.num_anchors = len(self.anchors)
        
        self.threshold = threshold
        
    def _build_strides(self) -> List[Tuple[int, int]]:
        strides = []
        for tensor_size in self.tensor_sizes:
            stride = self._build_stride(tensor_size)
            strides.append(stride)
            
        return strides
        
    def _build_stride(self, tensor_size: Tuple[int, int]) -> Tuple[int, int]:
        stride_x = self.img_size[0] / tensor_size[0]
        stride_y = self.img_size[1] / tensor_size[1]
        stride = (stride_x, stride_y)
        
        return stride
    
    def _build_anchor_maps(self) -> np.ndarray:
        anchor_maps = []
        for index in range(self.num_maps):
            stride = self.strides[index]
            tensor_size = self.tensor_sizes[index]
            anchor_map = self._build_anchor_map(stride, tensor_size, index)
            anchor_maps.extend(anchor_map)
            
        return np.array(anchor_maps)
        
    def _build_anchor_map(self, stride: Tuple[int, int], tensor_size: Tuple[int, int], index: int) -> List[List[int]]:
        stride_x = stride[0]
        stride_y = stride[1]
        x_grid = np.linspace(stride_x/2, self.img_size[0]-stride_x/2,
                             num=tensor_size[0])
        y_grid = np.linspace(stride_y/2, self.img_size[1]-stride_y/2,
                             num=tensor_size[1])
        x_centers, y_centers = np.meshgrid(x_grid, y_grid)
        centers = np.dstack((x_centers, y_centers)).reshape(-1,2)
        
        anchor_map = []
        
        for anchor_size in self.anchor_sizes[index]:
            for center in centers:
                anchor = self._build_anchor(center, anchor_size)
                anchor_map.append(anchor)
        
        return anchor_map

    def _build_left_top_maps(self):
        left_top_maps = []
        for index in range(self.num_maps):
            stride = self.strides[index]
            tensor_size = self.tensor_sizes[index]
            left_top_map = self._build_left_top_map(stride, tensor_size, index)
            left_top_maps.append(left_top_map)

        return np.vstack(left_top_maps)
            

    def _build_left_top_map(self, stride: Tuple[int, int], tensor_size: Tuple[int, int], index: int):
        stride_x = stride[0]
        stride_y = stride[1]

        horizon_grid = np.linspace(0, self.img_size[0]-stride_x, num=tensor_size[0])
        vertical_grid = np.linspace(0, self.img_size[1]-stride_y, num=tensor_size[1])
        left, top = np.meshgrid(horizon_grid, vertical_grid)
        left_top = np.dstack((left, top)).reshape(-1, 2)

        return np.vstack([left_top] * len(self.anchor_sizes[index]))

    
    def _build_anchor(self, center: Tuple[float, float], anchor_size: Tuple[int, int]) -> List[int]:
        width, height = anchor_size
        x_min = max(0, center[0] - width/2)
        y_min = max(0, center[1] - height/2)
        x_max = min(self.img_size[0], center[0] + width/2)
        y_max = min(self.img_size[1], center[1] + height/2)
        
        return [x_min, y_min, x_max, y_max]
    
    def compute_iou(self, bbox: List[int]) -> np.ndarray:
        inter_x_min = np.maximum(self.anchors[:,0], bbox[0])
        inter_y_min = np.maximum(self.anchors[:,1], bbox[1])
        inter_x_max = np.minimum(self.anchors[:,2], bbox[2]+bbox[0])
        inter_y_max = np.minimum(self.anchors[:,3], bbox[3]+bbox[1])
        
        inter_width = (inter_x_max - inter_x_min).clip(0)
        inter_height = (inter_y_max - inter_y_min).clip(0)
        
        inter_area = inter_width * inter_height
        
        anchors_width = self.anchors[:,2] - self.anchors[:,0]
        anchors_height = self.anchors[:,3] - self.anchors[:,1]
        anchors_area = anchors_width * anchors_height
        
        bbox_area = bbox[2] * bbox[3]
        
        ratios = inter_area / (anchors_area + bbox_area - inter_area)
        
        return ratios
    
    def scale_bbox(self, index: int, bbox: List[int]) -> np.ndarray:
        anchors_x = self.left_top_maps[index, 0]
        anchors_y = self.left_top_maps[index, 1]
        anchors_width = self.anchors[index,2] - self.anchors[index,0]
        anchors_height = self.anchors[index,3] - self.anchors[index,1]

        
        bbox_x = bbox[0] + bbox[2]/2
        bbox_y = bbox[1] + bbox[3]/2
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        stride_x, stride_y = self.get_stride_from_index(index)
        
        delta_x = (bbox_x - anchors_x) / stride_x
        delta_y = (bbox_y - anchors_y) / stride_y
        delta_width = np.log(bbox_width/anchors_width)
        delta_height = np.log(bbox_height/anchors_height)
        
        return np.array([[delta_x, delta_y, delta_width, delta_height]])
        
    def get_stride_from_index(self, index: int):
        ranges = []
        for i in range(len(self.tensor_sizes)):
            count = self.tensor_sizes[i][0] * self.tensor_sizes[i][1] * len(self.anchor_sizes[i])
            ranges.append(count)

        stride_index = 0
        for i in range(len(ranges)):
            stride_index = i
            if index < ranges[i]:
                break
        

        return self.strides[stride_index]
    
    def build_targets(self, bboxes: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        targets = np.zeros((self.num_anchors, self.channels,))
        obj_masks = np.zeros((self.num_anchors))
        no_obj_masks = np.ones((self.num_anchors))
        
        
        for bbox in bboxes:
            cla = bbox[0]+5
            size = bbox[1:]
            iou_ratios = self.compute_iou(size)
            object_mask = iou_ratios >= self.threshold
            max_iou_index = np.argmax(iou_ratios)
            targets[max_iou_index, :4] = self.scale_bbox(max_iou_index, size)
            targets[max_iou_index, 4:5] = 1.
            targets[max_iou_index, cla] = 1.
            obj_masks[max_iou_index] = 1
            no_obj_masks[object_mask] = 0
            
        return targets, obj_masks.astype(bool), no_obj_masks.astype(bool)

class Yolov3TargetGenerator:
    def __init__(self, tensor_sizes, img_size, anchor_sizes, num_cla):
        self.tensor_sizes = np.array(tensor_sizes)
        self.img_size = np.array(img_size)
        self.strides = self.img_size / self.tensor_sizes
        self.anchor_sizes = np.array(anchor_sizes)
        
        self.anchors_per_feature = [len(anchors) for anchors in self.anchor_sizes]
        self.feature_nums = len(self.tensor_sizes)
        self.channels = 5+num_cla
        self.threshold = 0.5
        
        
    def compute_iou(self, bbox):
        w = np.minimum(bbox[2], self.anchor_sizes[:, :, 0])
        h = np.minimum(bbox[3], self.anchor_sizes[:, :, 1])
        area = w * h
        b_area = bbox[2] * bbox[3]
        a_area = self.anchor_sizes[:, :, 0] * self.anchor_sizes[:, :, 1]

        iou = area / (a_area + b_area - area)

        return iou
    
    def get_mask_and_best(self, bbox):
        iou = self.compute_iou(bbox)
        mask = iou > self.threshold
        best = np.argmax(iou)
        feature_index = best // self.feature_nums
        anchor_index = best % self.anchors_per_feature[feature_index]
        
        return mask, feature_index, anchor_index
        
    def get_feature_map_index(self, center):
        scaled_y = (center[0] / self.img_size[0]) * self.tensor_sizes[:, 0]
        scaled_x = (center[1] / self.img_size[1]) * self.tensor_sizes[:, 1]
        
        index_y = scaled_y.astype(int)
        index_x = scaled_x.astype(int)
        
        target_x = scaled_x - index_x
        target_y = scaled_y - index_y
        
        return index_x, index_y, target_x, target_y
    
    def get_scaled_size(self, size):
        scaled_h = np.log(size[0] / (self.anchor_sizes[:, :, 0]))
        scaled_w = np.log(size[1] / (self.anchor_sizes[:, :, 1]))
        
        return scaled_w, scaled_h
    
    def build_target_matrix(self):
        targets = []
        for i in range(len(self.tensor_sizes)):
            anchor_num = self.anchors_per_feature[i]
            w = self.tensor_sizes[i][0]
            h = self.tensor_sizes[i][1]
            target = np.zeros((anchor_num, w, h, self.channels))
            targets.append(target)
            
        return targets
    
    def build_mask_matrix(self):
        obj_masks = []
        no_obj_masks = []
        
        for i in range(len(self.tensor_sizes)):
            anchor_num = self.anchors_per_feature[i]
            w = self.tensor_sizes[i][0]
            h = self.tensor_sizes[i][1]
            obj_mask = np.zeros((anchor_num, w, h)).astype(bool)
            no_obj_mask = np.ones_like(obj_mask).astype(bool)
            obj_masks.append(obj_mask)
            no_obj_masks.append(no_obj_mask)
            
        return obj_masks, no_obj_masks
    
    def get_target(self, target_x, target_y, target_w, target_h, cla, feature_index, anchor_index):
        target = [0] * self.channels
        target[0] = target_x[feature_index]
        target[1] = target_y[feature_index]
        target[2] = target_w[feature_index][anchor_index]
        target[3] = target_h[feature_index][anchor_index]
        target[4] = 1.
        target[cla+5] = 1.
        
        return target
    
    def get_anchor_mask(self, bbox):
        iou = self.compute_iou(bbox)
        mask = iou > self.threshold
        
        anchor_index = np.argmax(iou,1)
        
        return mask, anchor_index
    
    def build_target(self, target_x, target_y, target_w, target_h, cla, index, best_anchor):
        target = [0] * self.channels
        target[0] = target_x[index]
        target[1] = target_y[index]
        target[2] = target_w[index][best_anchor]
        target[3] = target_h[index][best_anchor]
        target[4] = 1.
        target[cla+5] = 1.
        
        return target
    
    def build_targets(self, bboxes):
        targets = self.build_target_matrix()
        obj_masks, no_obj_masks = self.build_mask_matrix()
        
        for bbox in bboxes:
            cla = bbox[0]
            bbox = bbox[1:]
            index_x, index_y, target_x, target_y = self.get_feature_map_index(bbox[:2])
            target_w, target_h = self.get_scaled_size(bbox[2:])
            #maskes, anchor_index = self.get_anchor_mask(bbox)
            
            
            maskes, feature_index, anchor_index = self.get_mask_and_best(bbox)
            
            best_index_x = index_x[feature_index]
            best_index_y = index_y[feature_index]
            
            #print(feature_index)
            #print(best_index_x)
            #print(best_index_y)
            #print(anchor_index)
            #print(maskes)
            target = self.get_target(target_x, target_y, target_w, target_h, cla, feature_index, anchor_index)
            targets[feature_index][anchor_index, best_index_x, best_index_y, :] = target
            obj_masks[feature_index][anchor_index, best_index_x, best_index_y] = 1
            no_obj_masks[feature_index][anchor_index, best_index_x, best_index_y] = 0
            
            
            for i in range(len(no_obj_masks)):
                mask = maskes[i]

                x = index_x[i]
                y = index_y[i]
                
                #best_anchor = anchor_index[i]
                
                #target = self.get_target(target_x, target_y, target_w, target_h, cla, i, best_anchor)
                
                #targets[i][best_anchor, x, y, :] = target
                #obj_masks[i][best_anchor, x, y] = 1
                #no_obj_masks[i][best_anchor, x, y] = 0
                no_obj_masks[i][mask, x, y] = 0
                
        targets = [target.reshape(-1, self.channels) for target in targets]
        obj_masks = [mask.reshape(-1, 1) for mask in obj_masks]
        no_obj_masks = [mask.reshape(-1, 1) for mask in no_obj_masks]
        
        targets = np.vstack(targets)
        obj_masks = np.vstack(obj_masks)
        no_obj_masks = np.vstack(no_obj_masks)
        
        return targets, obj_masks, no_obj_masks
