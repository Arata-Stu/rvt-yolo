import torch
import torch.nn as nn

from functools import partial
from omegaconf import DictConfig

from .network.backbone.darknet import CSPDarknet
from .network.backbone.pafpn import YOLOPAFPN
from .network.head.yolo_head import YOLOXHead

def build_darknet(backbone_config: DictConfig):
    return CSPDarknet(dep_mul=backbone_config.depth,
                      wid_mul=backbone_config.width,
                      input_dim=backbone_config.input_dim,
                      out_features=backbone_config.out_features,
                      depthwise=backbone_config.depthwise,
                      act=backbone_config.act)

def build_pafpn(fpn_config: DictConfig, in_channels):
    
    return YOLOPAFPN(
        depth=fpn_config.depth,  
        in_stages=fpn_config.in_stages,  
        in_channels=in_channels,  
        depthwise=fpn_config.depthwise,  
        act=fpn_config.act  
    )

def build_yolox_head(head_config: DictConfig, in_channels, strides):
    return YOLOXHead(
        num_classes=head_config.num_classes,   
        strides=strides,  
        in_channels=in_channels,  
        act=head_config.act, 
        depthwise=head_config.depthwise  
    )



