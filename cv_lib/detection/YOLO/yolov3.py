from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.basebone.basic_blocks.basic_conv import build_basic_block
from cv_lib.basebone.darknet import build_darknet_53_basic_conv

class Extractor(nn.Module):
    def __init__(self, in_channels:int):
        super(Extractor, self).__init__()
        self.block_1, self.block_2, self.block_3, self.block_4, self.block_5, self.block_6 = self.build_basebone(in_channels)


    def build_basebone(self, in_channels:int) -> List[nn.Module]:
        basebone = build_darknet_53_basic_conv(in_channels)
        return basebone.to_list()

    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        feature = self.block_1(tensor)
        feature = self.block_2(feature)
        feature = self.block_3(feature)
        output_3 = self.block_4(feature)
        output_2 = self.block_5(output_3)
        output_1 = self.block_6(output_2)

        return output_1, output_2, output_3

class YOLOV3Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, end=False):
        super(YOLOV3Layer, self).__init__()

        mid_channels_1 = out_channels * 2
        mid_channels_2 = out_channels * 4

        self.block_extract =  nn.Sequential(
            build_basic_block(in_channels, mid_channels_1, 1, 1, 0, activations=nn.LeakyReLU(inplace=True)),
            build_basic_block(mid_channels_1, mid_channels_2, 3, 1, 1, activations=nn.LeakyReLU(inplace=True)),
            build_basic_block(mid_channels_2, mid_channels_1, 1, 1, 0, activations=nn.LeakyReLU(inplace=True)),
            build_basic_block(mid_channels_1, mid_channels_2, 3, 1, 1, activations=nn.LeakyReLU(inplace=True)),
            build_basic_block(mid_channels_2, mid_channels_1, 1, 1, 0, activations=nn.LeakyReLU(inplace=True)),
        )

        self.block_smooth: Optional[nn.Module] = None
        if not end:
            self.block_smooth = build_basic_block(mid_channels_1, out_channels, 1, 1, 0, activations=nn.LeakyReLU(inplace=True))
            
        self.init_params()
            
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tensor: torch.Tensor) -> List[Optional[torch.Tensor]]:
        feature = self.block_extract(tensor)

        if self.block_smooth is not None:
            feature_smooth = self.block_smooth(feature)
            feature_upsample = F.interpolate(feature_smooth, scale_factor=2)

            return feature, feature_upsample
        else:
            return feature, None

class YOLOV3Head(nn.Module):
    def __init__(self, in_channels, num_cla):
        super(YOLOV3Head, self).__init__()
        mid_channels = in_channels * 2
        self.block = build_basic_block(in_channels, mid_channels, 3, 1, 1, activations=nn.LeakyReLU(inplace=True))
        self.head = nn.Conv2d(mid_channels, (5+num_cla) * 3, 1, 1, 0)
        
    
    def forward(self, tensor: torch.Tensor):
        output = self.block(tensor)
        output = self.head(output)
        
        batch_size, channels, width, height = output.size()
        channels = channels // 3
        output = output.view(batch_size, -1, channels, width, height).permute(0, 1, 3, 4, 2).contiguous()



        #print(output[0, :, 0, 0])
        if not self.training:
            output_xy = torch.sigmoid(output[:, :, :, :, :2])
            output_wh = output[:, :, :, :, 2:4]
            output_obj = torch.sigmoid(output[:, :, :, :, 4:5])
            output_cla = torch.sigmoid(output[:, :, :, :, 5:])
            
            output = torch.cat([output_xy, output_wh, output_obj, output_cla], dim=4)
        

        return output

class YOLOV3(nn.Module):
    def __init__(self, in_channels:int, num_cla:int):
        super(YOLOV3, self).__init__()
        self.extractor = Extractor(in_channels)

        self.block_13x13 = YOLOV3Layer(1024, 256)
        self.block_26x26 = YOLOV3Layer(768, 128)
        self.block_52x52 = YOLOV3Layer(384, 64, end=True)

        self.head_13x13 = YOLOV3Head(512, num_cla)
        self.head_26x26 = YOLOV3Head(256, num_cla)
        self.head_52x52 = YOLOV3Head(128, num_cla)

    def forward(self, tensor: torch.Tensor):
        feature_13x13, feature_26x26, feature_52x52 = self.extractor(tensor)

        feature_13x13, feature_13x13_upsample = self.block_13x13(feature_13x13)
        feature_26x26 = torch.cat([feature_26x26, feature_13x13_upsample], dim=1)
        feature_26x26, feature_26x26_upsample = self.block_26x26(feature_26x26)
        feature_52x52 = torch.cat([feature_52x52, feature_26x26_upsample], dim=1)
        feature_52x52, _ = self.block_52x52(feature_52x52)

        output_13x13 = self.head_13x13(feature_13x13)
        output_26x26 = self.head_26x26(feature_26x26)
        output_52x52 = self.head_52x52(feature_52x52)
        
        
        if not self.training:
            return output_13x13, output_26x26, output_52x52
        
        batch_size, _, _, _, channels = output_13x13.size()
        
        output_13x13 = output_13x13.view(batch_size, -1, channels)
        output_26x26 = output_26x26.view(batch_size, -1, channels)
        output_52x52 = output_52x52.view(batch_size, -1, channels)
        
        output = torch.cat([output_13x13, output_26x26, output_52x52], dim=1)
        
        return output
        #return output_13x13, output_26x26, output_52x52

def build_yolov3(in_channels, num_cla):
    return YOLOV3(in_channels, num_cla)


        
        
        
