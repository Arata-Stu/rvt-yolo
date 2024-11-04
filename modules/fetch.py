import pytorch_lightning as pl
from omegaconf import DictConfig

from .model import DNNModule, RNNModule, SSMModule
from modules.data.data_module import DataModule

def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    model_str = config.model.type
    if model_str == 'dnn':
        return DNNModule(config)
    elif model_str == 'rnn':
        return RNNModule(config)
    elif model_str == 'smm':
        return SSMModule(config)
    
    raise NotImplementedError

def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    return DataModule(config)