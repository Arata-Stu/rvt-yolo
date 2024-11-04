import torch.nn as nn
from functools import partial
from omegaconf import DictConfig

from models.detection.yolox_lstm.yolox_lstm import Darknet_LSTM
from .yolox.yolox import build_darknet, build_pafpn, build_yolox_head
from models.detection.RVT.models.detection.recurrent_backbone import build_recurrent_backbone

from models.detection.yolox.utils.boxes import postprocess

def build_backbone(backbone_config: DictConfig):
    name = backbone_config.name
    if name == 'darknet':
        print('darknet')
        backbone = build_darknet(backbone_config=backbone_config)
    elif name == 'darknet_lstm':
        print('darknet')
        print('LSTM')
        backbone = Darknet_LSTM(backbone_config=backbone_config)
    elif name == 'MaxViTRNN' or name == 'MaxViTRNN-SSM':
        print('RVT')
        backbone = build_recurrent_backbone(backbone_cfg=backbone_config)
    else:
        NotImplementedError
    
    return backbone

def build_fpn(fpn_config: DictConfig, in_channels):
    if fpn_config.name == 'pafpn':
        print('PAFPN')
        fpn = build_pafpn(fpn_config=fpn_config, in_channels=in_channels)
    else:
        NotImplementedError
    
    return fpn

def build_head(head_config: DictConfig, in_channels, strides):
    if head_config.name == 'yolox':
        print('YOLOX Head')
        head = build_yolox_head(head_config=head_config, in_channels=in_channels, strides=strides)
        head.initialize_biases(prior_prob=0.01)
    else:
        NotImplementedError

    return head

class DNNDetectionModel(nn.Module):
    def __init__(self, model_config:DictConfig):
        super().__init__()

        bb_config = model_config.backbone
        fpn_config = model_config.fpn
        head_config = model_config.head
        postprocess_config = model_config.postprocess

        assert bb_config.depth == fpn_config.depth 
       
        backbone = build_backbone(backbone_config=bb_config)

        in_channels = backbone.get_stage_dims(fpn_config.in_stages)
        fpn = build_fpn(fpn_config=fpn_config, in_channels=in_channels)

        strides = backbone.get_strides(fpn_config.in_stages)
        head = build_head(head_config=head_config, in_channels=in_channels, strides=strides)

        print('stage_dim', in_channels)
        print('strides', strides)
        self.backbone = backbone
        self.fpn = fpn
        self.head=head
        self.post_process = partial(postprocess,
                                    num_classes=head_config.num_classes,
                                    conf_thre=postprocess_config.conf_thre,
                                    nms_thre=postprocess_config.nms_thre,
                                    class_agnostic=False)

    def forward(self, x, targets=None):
        backbone_features = self.backbone(x)
        fpn_outs = self.fpn(backbone_features)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            outputs = outputs["total_loss"]
        else:
            predictions = self.head(fpn_outs)
            outputs = self.post_process(prediction=predictions)
      
        return outputs
    

class RNNDetectionModel(nn.Module):
    def __init__(self, model_config:DictConfig):
        super().__init__()

        bb_config = model_config.backbone
        fpn_config = model_config.fpn
        head_config = model_config.head
        postprocess_config = model_config.postprocess

        
        backbone = build_backbone(backbone_config=bb_config)
        in_channels = backbone.get_stage_dims(fpn_config.in_stages)
        fpn = build_fpn(fpn_config=fpn_config, in_channels=in_channels)

        strides = backbone.get_strides(fpn_config.in_stages)
        head = build_head(head_config=head_config, in_channels=in_channels, strides=strides)

        print('stage_dim', in_channels)
        print('strides', strides)
            
        self.backbone = backbone
        self.fpn = fpn
        self.head=head
        self.post_process = partial(postprocess,
                                    num_classes=head_config.num_classes,
                                    conf_thre=postprocess_config.conf_thre,
                                    nms_thre=postprocess_config.nms_thre,
                                    class_agnostic=False)

    def forward_backbone(self,
                         x,
                         previous_states = None,
                         token_mask = None):
    
        backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states
    
    def forward_detect(self,
                       backbone_features,
                       targets = None):
        
        fpn_features = self.fpn(backbone_features)
        if self.training:
            assert targets is not None    
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_features, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            outputs = outputs["total_loss"]
            return outputs
        else:
            predictions = self.head(fpn_features)
            outputs = self.post_process(prediction=predictions)
        
            return outputs

    def forward(self, 
                x, 
                previous_states = None,
                targets = None):
        backbone_features, states = self.forward_backbone(x, previous_states)
        
        outputs = self.forward_detect(backbone_features, targets)
        return outputs, states