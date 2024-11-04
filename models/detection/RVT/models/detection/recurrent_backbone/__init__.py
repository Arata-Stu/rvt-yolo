from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        print('LSTM')
        return MaxViTRNNDetector(backbone_cfg)
    if name == 'MaxViTRNN-SSM':
        print('SSM')
        return MaxViTRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
