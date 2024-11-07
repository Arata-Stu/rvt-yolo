import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from omegaconf import DictConfig
from warnings import warn

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

    def on_validation_epoch_end(self):
        self.run_eval(self.val_evaluator, mode="val")

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

    def on_test_epoch_end(self):
        self.run_eval(self.test_evaluator, mode="test")

    def run_eval(self, evaluator, mode, batch_size, hw_tuple):
        """評価のための共通関数（分散処理なし）"""
        if evaluator is None:
            warn(f'Evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        
        assert batch_size is not None, "Batch size is None"
        assert hw_tuple is not None, "Image height and width are not set"
        
        # 評価バッファにデータがあるか確認
        if evaluator.has_data():
            # 画像サイズを指定してメトリクスを評価
            metrics = evaluator.evaluate_buffer(img_height=hw_tuple[0], img_width=hw_tuple[1])
            assert metrics is not None, "Evaluation metrics are None"

            # ログ用のディクショナリを作成
            prefix = f'{mode}/'
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError(f"Unsupported type for metric {k}: {type(v)}")
                    
                # 値がスカラーであることを確認し、デバイスに送る
                assert value.ndim == 0, f'Metric {k} must be a scalar, got {value.ndim} dimensions'
                log_dict[f'{prefix}{k}'] = value.to(self.device)

            # メトリクスのロギング
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size)
            
            # グローバルステップを使ったメトリクスログ（WandBなどの場合）
            if self.trainer.is_global_zero:
                add_hack = 2
                step = self.trainer.global_step + add_hack
                self.logger.log_metrics(metrics=log_dict, step=step)

            # 評価バッファをリセット
            evaluator.reset_buffer()
        else:
            warn(f'Evaluator has no data in {mode=}', UserWarning, stacklevel=2)
    
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
