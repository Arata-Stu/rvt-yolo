import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from omegaconf import DictConfig
from warnings import warn

from models.detection.build import RNNDetectionModel
from modules.utils.rnn_state import RNNStates
from utils.eval.prophesee.evaluator import PropheseeEvaluator
from utils.eval.prophesee.io.box_loading import to_prophesee
from models.detection.yolox.utils.boxes import postprocess


class RNNModule(pl.LightningModule):

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
        self.model = RNNDetectionModel(model_config=full_config.model)

        self.post_process = partial(postprocess,
                                    num_classes=full_config.model.head.num_classes,
                                    conf_thre=full_config.model.postprocess.conf_thre,
                                    nms_thre=full_config.model.postprocess.nms_thre,
                                    class_agnostic=False)


        self.train_rnn_state = RNNStates()
        self.test_rnn_state = RNNStates()
        self.val_rnn_state = RNNStates()

       
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

        # バッチデータ取得
        sequence_events = batch['events']  # [batch, sequence_len, ch, h, w]
        sequence_targets = batch['labels']  # [batch, sequence_len, num, bbox]
        is_first_sample = batch['is_first_sample']  # [batch, (h,w)]
        sequence_is_padded_mask = batch['is_padded_mask']  # [batch, sequence_len]
        sequence_timestamps = batch['timestamps']
        # RNN状態の初期化
        worker_id = 0
        self.train_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.train_rnn_state.get_states(worker_id=worker_id)

        batch_size, sequence_length, ch, height, width = sequence_events.shape

        all_saved_data = []
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)

        for t in range(sequence_length):
            # 各時系列ステップのデータ取得
            events = sequence_events[:, t, :, :, :].to(dtype=self.dtype)  
            targets = sequence_targets[:, t, :, :].to(dtype=self.dtype)  
            token_mask = sequence_is_padded_mask[:, t]  
            time = sequence_timestamps[:, t]
            targets.requires_grad = False

            # RNNバックボーンへの入力
            backbone_features, states = self.model.forward_backbone(
                x=events, previous_states=prev_states, token_mask=None
            )
            prev_states = states

            # ラベルが存在するバッチインデックスの取得
            valid_batch_indices = [i for i, batch in enumerate(targets) if (batch != padding).any()]

            if valid_batch_indices:
                # データの保存
                step_data = {
                    "step": t,
                    "valid_batch_indices": valid_batch_indices,
                    "selected_targets": targets[valid_batch_indices],
                    "selected_backbone_features": {key: feature[valid_batch_indices] for key, feature in backbone_features.items()}
                }
                all_saved_data.append(step_data)

        self.train_rnn_state.save_states_and_detach(worker_id=worker_id, states=prev_states)

        # バッチ化データの生成
        if all_saved_data:  # 有効なデータがあるか確認
            batched_backbone_features = {key: [] for key in all_saved_data[0]['selected_backbone_features'].keys()}
            batched_targets = []

            for data in all_saved_data:
                for key, feature in data['selected_backbone_features'].items():
                    batched_backbone_features[key].append(feature)
                batched_targets.append(data['selected_targets'])

            batched_backbone_features = {key: torch.cat(features, dim=0) for key, features in batched_backbone_features.items()}
            batched_targets = torch.cat(batched_targets, dim=0)

            loss = self.model.forward_detect(
                backbone_features=batched_backbone_features,
                targets=batched_targets
            )

            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss
        else:
            print("No valid data for detection.")
            return None



    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].mean()
        self.log('epoch_train_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # モデルを評価モードに設定
        self.model.eval()
        model_to_eval = self.model
        model_to_eval.to(self.device)

        # データ取得
        sequence_events = batch['events']  # [batch, sequence_len, ch, h, w]
        sequence_targets = batch['labels']  # [batch, sequence_len, num, bbox]
        is_first_sample = batch['is_first_sample']  # [batch, (h,w)]
        sequence_is_padded_mask = batch['is_padded_mask']  # [batch, sequence_len]
        sequence_timestamps = batch['timestamps']  # [batch, sequence_len]

        # 初期状態のリセット
        worker_id = 0
        self.val_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.val_rnn_state.get_states(worker_id=worker_id)

        batch_size, sequence_length, ch, height, width = sequence_events.shape

        all_saved_data = []
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)

        for t in range(sequence_length):
            # 各時系列ステップのデータ取得
            events = sequence_events[:, t, :, :, :].to(dtype=self.dtype)
            targets = sequence_targets[:, t, :, :].to(dtype=self.dtype)
            token_mask = sequence_is_padded_mask[:, t]
            time = sequence_timestamps[:, t]
            targets.requires_grad = False
            
            # 時系列データをRNNに渡す
            backbone_features, states = self.model.forward_backbone(
                x=events, previous_states=prev_states, token_mask=token_mask
            )
            prev_states = states
            
            # ラベルが存在するバッチインデックスの取得
            valid_batch_indices = [i for i, batch in enumerate(targets) if (batch != padding).any()]

            if valid_batch_indices:
                # タイムスタンプ付きでデータを保存
                step_data = {
                    "step": t,
                    "valid_batch_indices": valid_batch_indices,
                    "selected_targets": targets[valid_batch_indices],
                    "selected_backbone_features": {key: feature[valid_batch_indices] for key, feature in backbone_features.items()},
                    "time": time[valid_batch_indices]  # タイムスタンプも保存
                }
                all_saved_data.append(step_data)

        self.val_rnn_state.save_states_and_detach(worker_id=worker_id, states=prev_states)

        if all_saved_data:  # 有効なデータがあるか確認
            batched_backbone_features = {key: [] for key in all_saved_data[0]['selected_backbone_features'].keys()}
            batched_targets = []
            batched_times = []

            for data in all_saved_data:
                for key, feature in data['selected_backbone_features'].items():
                    batched_backbone_features[key].append(feature)
                batched_targets.append(data['selected_targets'])
                batched_times.append(data['time'])  # タイムスタンプもバッチに追加

            batched_backbone_features = {key: torch.cat(features, dim=0) for key, features in batched_backbone_features.items()}
            batched_targets = torch.cat(batched_targets, dim=0)
            batched_times = torch.cat(batched_times, dim=0)  # タイムスタンプのバッチ化

            preds = self.model.forward_detect(
                backbone_features=batched_backbone_features
            )

            processed_preds = self.post_process(prediction=preds)

            # Prophesee形式に変換
            loaded_labels_proph, yolox_preds_proph = to_prophesee(
                loaded_label_tensor=batched_targets,
                label_timestamps=batched_times,  # タイムスタンプを含めて渡す
                yolox_pred_list=processed_preds
            )

            # 評価器にラベルと予測を追加
            if self.started_training:
                self.val_evaluator.add_labels(loaded_labels_proph)
                self.val_evaluator.add_predictions(yolox_preds_proph)

        return  # 返すべきものがあれば適宜追記

    def on_validation_epoch_end(self):
        if self.started_training:
            if self.val_evaluator.has_data():
                metrics = self.val_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    self.log(f'val_{k}', v, on_epoch=True, prog_bar=True, logger=True)
                self.val_evaluator.reset_buffer()

    def test_step(self, batch, batch_idx):
        # モデルを評価モードに設定
        self.model.eval()
        model_to_eval = self.model
        model_to_eval.to(self.device)

        # データ取得
        sequence_events = batch['events']  # [batch, sequence_len, ch, h, w]
        sequence_targets = batch['labels']  # [batch, sequence_len, num, bbox]
        is_first_sample = batch['is_first_sample']  # [batch, (h,w)]
        sequence_is_padded_mask = batch['is_padded_mask']  # [batch, sequence_len]
        sequence_timestamps = batch['timestamps']  # [batch, sequence_len]

        # 初期状態のリセット
        worker_id = 0
        self.test_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.test_rnn_state.get_states(worker_id=worker_id)

        batch_size, sequence_length, ch, height, width = sequence_events.shape

        all_saved_data = []
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)

        for t in range(sequence_length):
            # 各時系列ステップのデータ取得
            events = sequence_events[:, t, :, :, :].to(dtype=self.dtype)
            targets = sequence_targets[:, t, :, :].to(dtype=self.dtype)
            token_mask = sequence_is_padded_mask[:, t]
            time = sequence_timestamps[:, t]
            targets.requires_grad = False
            
            # 時系列データをRNNに渡す
            backbone_features, states = self.model.forward_backbone(
                x=events, previous_states=prev_states, token_mask=token_mask
            )
            prev_states = states
            
            # ラベルが存在するバッチインデックスの取得
            valid_batch_indices = [i for i, batch in enumerate(targets) if (batch != padding).any()]

            if valid_batch_indices:
                # タイムスタンプ付きでデータを保存
                step_data = {
                    "step": t,
                    "valid_batch_indices": valid_batch_indices,
                    "selected_targets": targets[valid_batch_indices],
                    "selected_backbone_features": {key: feature[valid_batch_indices] for key, feature in backbone_features.items()},
                    "time": time[valid_batch_indices]  # タイムスタンプも保存
                }
                all_saved_data.append(step_data)

        self.test_rnn_state.save_states_and_detach(worker_id=worker_id, states=prev_states)

        if all_saved_data:  # 有効なデータがあるか確認
            batched_backbone_features = {key: [] for key in all_saved_data[0]['selected_backbone_features'].keys()}
            batched_targets = []
            batched_times = []

            for data in all_saved_data:
                for key, feature in data['selected_backbone_features'].items():
                    batched_backbone_features[key].append(feature)
                batched_targets.append(data['selected_targets'])
                batched_times.append(data['time'])  # タイムスタンプもバッチに追加

            batched_backbone_features = {key: torch.cat(features, dim=0) for key, features in batched_backbone_features.items()}
            batched_targets = torch.cat(batched_targets, dim=0)
            batched_times = torch.cat(batched_times, dim=0)  # タイムスタンプのバッチ化

            preds = self.model.forward_detect(
                backbone_features=batched_backbone_features
            )

            processed_preds = self.post_process(prediction=preds)

            # Prophesee形式に変換
            loaded_labels_proph, yolox_preds_proph = to_prophesee(
                loaded_label_tensor=batched_targets,
                label_timestamps=batched_times,  # タイムスタンプを含めて渡す
                yolox_pred_list=processed_preds
            )

            # 評価器にラベルと予測を追加
            if self.started_training:
                self.test_evaluator.add_labels(loaded_labels_proph)
                self.test_evaluator.add_predictions(yolox_preds_proph)

        return  # 返すべきものがあれば適宜追記

    def on_test_epoch_end(self):
        if self.started_training:
            if self.test_evaluator.has_data():
                metrics = self.test_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    self.log(f'test_{k}', v, on_epoch=True, prog_bar=True, logger=True)
                self.test_evaluator.reset_buffer()

    def run_eval(self, evaluator, mode):
        """評価のための共通関数（分散処理なし）"""
        if evaluator is None:
            warn(f'Evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        
        # 評価バッファにデータがあるか確認
        if evaluator.has_data():
            # 画像サイズを指定してメトリクスを評価
            metrics = evaluator.evaluate_buffer(img_height=self.height, img_width=self.width)
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
            self.log_dict(log_dict, on_step=False, on_epoch=True)
            
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



