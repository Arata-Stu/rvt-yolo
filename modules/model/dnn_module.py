import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from omegaconf import DictConfig

from models.detection.build import DNNDetectionModel
from utils.eval.prophesee.evaluator import PropheseeEvaluator
from utils.eval.prophesee.io.box_loading import to_prophesee
from models.detection.yolox.utils.boxes import postprocess


class DNNModule(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config
        self.validation_scores = []  # バリデーションスコアを保存するリスト
        self.test_scores = []

        if self.full_config.dataset.name == "gen1":
            from data.dataset.genx.classes import  GEN1_CLASSES as CLASSES
            self.height, self.width = 240, 304
        elif self.full_config.dataset.name == "gen4":
            from data.dataset.genx.classes import GEN4_CLASSES as CLASSES
            self.height, self.width = 360, 640
        
        self.classes = CLASSES  # クラスを保持
        self.model = DNNDetectionModel(model_config=full_config.model)

        self.post_process = partial(postprocess,
                                    num_classes=full_config.model.head.num_classes,
                                    conf_thre=full_config.model.postprocess.conf_thre,
                                    nms_thre=full_config.model.postprocess.nms_thre,
                                    class_agnostic=False)


       
    def setup(self, stage):
        self.started_training = True
        
        if stage == 'fit':
            self.started_training = False
            self.val_evaluator = PropheseeEvaluator(dataset=self.full_config.dataset.name, 
                                                downsample_by_2=self.full_config.dataset.val.downsample_by_factor_2)
        elif stage == 'test':
            self.test_evaluator = PropheseeEvaluator(dataset=self.full_config.dataset.name, 
                                                downsample_by_2=self.full_config.dataset.test.downsample_by_factor_2)
        
        
    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        self.started_training = True
        self.model.train()
        imgs = batch['events'][:, 0].to(dtype=self.dtype)  
        targets = batch['labels'][:, 0].to(dtype=self.dtype)
        
        targets.requires_grad = False
        
        loss = self(imgs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].mean()
        self.log('epoch_train_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        
        self.model.eval()
        model_to_eval = self.model

        model_to_eval.to(self.device)
        imgs = batch['events'][:, 0].to(dtype=self.dtype)  
        targets = batch['labels'][:, 0].to(dtype=self.dtype)
        timestamps = batch['timestamps'][:, -1]

        targets.requires_grad = False
        
        preds = model_to_eval(imgs)

        processed_preds = self.post_process(prediction=preds)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(loaded_label_tensor=targets, 
                                                              label_timestamps=timestamps, 
                                                              yolox_pred_list=processed_preds)

        if self.started_training:
            self.val_evaluator.add_labels(loaded_labels_proph)
            self.val_evaluator.add_predictions(yolox_preds_proph)

        
        return 

    def on_validation_epoch_end(self):
        if self.started_training:
            if self.val_evaluator.has_data():
                metrics = self.val_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    self.log(f'val_{k}', v, on_epoch=True, prog_bar=True, logger=True)
                self.val_evaluator.reset_buffer()

    def test_step(self, batch, batch_idx):
        self.model.eval()
        model_to_eval = self.model

        model_to_eval.to(self.device)
        imgs = batch['events'][:, 0].to(dtype=self.dtype)  
        targets = batch['labels'][:, 0].to(dtype=self.dtype)
        timestamps = batch['timestamps'][:, -1]
        targets.requires_grad = False
        
        preds = model_to_eval(imgs)

        processed_preds = self.post_process(prediction=preds)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(loaded_label_tensor=targets, 
                                                              label_timestamps=timestamps, 
                                                              yolox_pred_list=processed_preds)


        if self.started_training:
            self.test_evaluator.add_labels(loaded_labels_proph)
            self.test_evaluator.add_predictions(yolox_preds_proph)

        return 

    def on_test_epoch_end(self):
        if self.started_training:
            if self.test_evaluator.has_data():
                metrics = self.test_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    self.log(f'test_{k}', v, on_epoch=True, prog_bar=True, logger=True)
                self.test_evaluator.reset_buffer()
        
        
    def configure_optimizers(self):
        lr = self.full_config.experiment.training.learning_rate
        weight_decay = self.full_config.experiment.training.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.full_config.experiment.training.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}



