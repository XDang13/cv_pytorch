
from typing import Union
import json
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from random import sample

from cv_lib.utils.object_detection.anchors_map import Yolov3TargetGenerator, AnchorMap

class ObjectDetectionDataset(Dataset):
    def __init__(self, img_path: str, annotation_path: str, file_path: str, anchor_maps: Union[AnchorMap, Yolov3TargetGenerator]):
        super(ObjectDetectionDataset, self).__init__()
        self.img_path = img_path
        self.files = self.read_json(file_path)
        self.generator = anchor_maps
        self.transform = transforms.Compose([
            transforms.Resize((416,416)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        file = self.files[index]
        info = self.read_json(file)

        img_file = "{}{}.jpg".format(self.img_path, info["name"])
        img = Image.open(img_file)
        img_tensor = self.transform(img)

        bboxes = info["bboxes"]
        targets, obj_masks, no_obj_masks = self.generator.build_targets(bboxes)
        
        targets = torch.from_numpy(targets).float()
        obj_masks = torch.from_numpy(obj_masks)
        no_obj_masks = torch.from_numpy(no_obj_masks)

        #targets = [torch.from_numpy(target).float() for target in targets]
        
        #obj_masks = [torch.from_numpy(obj_mask) for obj_mask in obj_masks]

        #no_obj_masks = [torch.from_numpy(no_obj_mask) for no_obj_mask in no_obj_masks]

        return img_tensor, targets, obj_masks, no_obj_masks

    def __len__(self):
        return len(self.files)

    def read_json(self, file: str):
        with open(file, "r") as f:
            info = json.load(f)

        return info
