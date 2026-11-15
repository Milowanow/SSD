# vgg_ssd.py
import torch
import torch.nn as nn
import torchvision

from ObjectDetectionProblemFromScratch.prior_boxes import prior_boxes

import logging
logger = logging.getLogger(__name__)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        #1
        self.block_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #2
        self.block_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #3
        self.block_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #4
        self.block_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2,  ceil_mode=True)

        #5
        self.block_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block_5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        #6
        self.block_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride = 1, padding=6, dilation=6),
            nn.ReLU(inplace=True)
        )

        #7
        self.block_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        #8
        self.block_8_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block_8_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride = 2, padding=1),
            nn.ReLU(inplace=True)
        )

        #9
        self.block_9_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block_9_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 2, padding=1),
            nn.ReLU(inplace=True)
        )

        #10
        self.block_10_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block_10_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )

        #11
        self.block_11_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block_11_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding=0),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        #1
        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.pool_1(x)
        
        #2
        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.pool_2(x)
        
        #3
        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_3_3(x)
        x = self.pool_3(x)
        
        #4
        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.block_4_3(x)
        feat_38x38 = x 
        x = self.pool_4(x)
        
        #5
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.block_5_3(x)
        x = self.pool_5(x)
        
        #6
        x = self.block_6(x)
        
        #7
        x = self.block_7(x)
        feat_19x19 = x 
        
        
        #8
        x = self.block_8_1(x)
        x = self.block_8_2(x)
        feat_10x10 = x

        #9
        x = self.block_9_1(x)
        x = self.block_9_2(x)
        feat_5x5 = x
       
        #10
        x = self.block_10_1(x)
        x = self.block_10_2(x)
        feat_3x3 = x
        
        #11
        x = self.block_11_1(x)
        x = self.block_11_2(x)
        feat_1x1 = x

        logger.debug(f"feat_38: {feat_38x38.shape}")
        logger.debug(f"feat_19: {feat_19x19.shape}")
        logger.debug(f"feat_10: {feat_10x10.shape}")
        logger.debug(f"feat_5:  {feat_5x5.shape}")
        logger.debug(f"feat_3:  {feat_3x3.shape}")
        logger.debug(f"feat_1:  {feat_1x1.shape}")

        return [feat_38x38, feat_19x19, feat_10x10, feat_5x5, feat_3x3, feat_1x1]

class PredictionConvolution(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.loc = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=num_anchors * 4, 
            kernel_size=3, 
            padding=1
        )
        
        self.conf = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        loc = self.loc(x)
        conf = self.conf(x)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        conf = conf.permute(0, 2, 3, 1).contiguous()
        
        loc = loc.view(loc.size(0), -1, 4) 
        conf = conf.view(conf.size(0), -1, self.num_classes)
        
        return loc, conf


class SSD_VGG16(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes
        
        self.vgg = VGG16()

        self.pred_38 = PredictionConvolution(in_channels=512, num_anchors=4, num_classes=num_classes)
        self.pred_19 = PredictionConvolution(in_channels=1024, num_anchors=6, num_classes=num_classes)
        self.pred_10 = PredictionConvolution(in_channels=512, num_anchors=6, num_classes=num_classes)
        self.pred_5 = PredictionConvolution(in_channels=256, num_anchors=6, num_classes=num_classes)
        self.pred_3 = PredictionConvolution(in_channels=256, num_anchors=4, num_classes=num_classes)
        self.pred_1 = PredictionConvolution(in_channels=256, num_anchors=4, num_classes=num_classes)
        self.priors = prior_boxes(config)


    def forward(self, x):
        features = self.vgg(x) 

        feat_38, feat_19, feat_10, feat_5, feat_3, feat_1 = features

        loc_38, conf_38 = self.pred_38(feat_38)
        loc_19, conf_19 = self.pred_19(feat_19)
        loc_10, conf_10 = self.pred_10(feat_10)
        loc_5,  conf_5  = self.pred_5(feat_5)
        loc_3,  conf_3  = self.pred_3(feat_3)
        loc_1,  conf_1  = self.pred_1(feat_1)
        
        locs = torch.cat([loc_38, loc_19, loc_10, loc_5, loc_3, loc_1], dim=1)
        confs = torch.cat([conf_38, conf_19, conf_10, conf_5, conf_3, conf_1], dim=1)
        
        return locs, confs, self.priors       