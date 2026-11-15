# vgg_ssd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ObjectDetectionProblemFromScratch.prior_boxes import prior_boxes

import logging
logger = logging.getLogger(__name__)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = stride),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = 1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Identity()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride = stride),
                nn.BatchNorm2d(out_channels),
            )
            
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.block(x)
        x = x_shortcut + x
        x = F.relu(x, inplace=True)
        return x

class ExtraBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0, stride = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )
    def forward(self, x):
        return self.block(x)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride = 2),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNetBlock(64, 64, downsample=False, stride=1),
            ResNetBlock(64, 64, downsample=False, stride=1),
            ResNetBlock(64, 128, downsample=True, stride=2),
            ResNetBlock(128, 128, downsample=False, stride=1),
            ResNetBlock(128, 256, downsample=True, stride=2),
            ResNetBlock(256, 256, downsample=False, stride=1),
        )

        self.res_net_block_1 = nn.Sequential(
            ResNetBlock(256, 512, downsample=True, stride=2),
            ResNetBlock(512, 512, downsample=False, stride=1)
        )
        self.res_net_block_2 = nn.Sequential(
            ResNetBlock(512, 1024, downsample=True, stride=2),
            ResNetBlock(1024, 1024, downsample=False, stride=1),
        )
        self.extra_block_1 = ExtraBlock()
        self.extra_block_2 = ExtraBlock()


    def forward(self, x):
        x = self.backbone(x)
        feat_1 = x #19x19

        x = self.res_net_block_1(x)
        feat_2 = x #10x10

        x = self.res_net_block_2(x)
        feat_3 = x #5x5

        x = self.extra_block_1(x)
        feat_4 = x #3x3

        x = self.extra_block_2(x)
        feat_5 = x #2x2

        
        logger.debug(f"feat_1: {feat_1.shape}")
        logger.debug(f"feat_2: {feat_2.shape}")
        logger.debug(f"feat_3: {feat_3.shape}")
        logger.debug(f"feat_4: {feat_4.shape}")
        logger.debug(f"feat_5: {feat_5.shape}")
        
        return [feat_1, feat_2, feat_3, feat_4, feat_5]
        

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


class SSD_ResNet18(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes
        
        self.res_net_18 = ResNet18()

        self.pred_1 = PredictionConvolution(in_channels=256, num_anchors=4, num_classes=num_classes)
        self.pred_2 = PredictionConvolution(in_channels=512, num_anchors=6, num_classes=num_classes)
        self.pred_3 = PredictionConvolution(in_channels=1024, num_anchors=6, num_classes=num_classes)
        self.pred_4 = PredictionConvolution(in_channels=1024, num_anchors=6, num_classes=num_classes)
        self.pred_5 = PredictionConvolution(in_channels=1024, num_anchors=4, num_classes=num_classes)
        self.priors = prior_boxes(config)


    def forward(self, x):
        features = self.res_net_18(x) 

        feat_1, feat_2, feat_3, feat_4, feat_5 = features

        loc_1, conf_1 = self.pred_1(feat_1)
        loc_2, conf_2 = self.pred_2(feat_2)
        loc_3, conf_3 = self.pred_3(feat_3)
        loc_4, conf_4  = self.pred_4(feat_4)
        loc_5, conf_5  = self.pred_5(feat_5)

        
        locs = torch.cat([loc_5, loc_4, loc_3, loc_2, loc_1], dim=1)
        confs = torch.cat([conf_5, conf_4, conf_3, conf_2, conf_1], dim=1)
        
        return locs, confs, self.priors       