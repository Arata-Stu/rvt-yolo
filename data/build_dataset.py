from functools import partial
from torchvision import transforms
from omegaconf import DictConfig
from utils.load import get_dataloading_hw
from .dataset import build_genx_dataset
from .data_utils.transform import EventPadderTransform, LabelPaddingTransform, YOLOXFormatter, RandomSpatialAugmentor

def build_dataset(full_config: DictConfig, mode='train'):
    # データセット設定の読み込み
    dataset_cfg = full_config.dataset
    target_hw = get_dataloading_hw(hw=dataset_cfg.target_size)

    # データ拡張の設定（トレーニング専用）
    if mode == 'train':
        aug_cfg = dataset_cfg.train.data_augmentation
        random_augmentation = RandomSpatialAugmentor(
            h_flip_prob=aug_cfg.prob_hflip,
            rotation_prob=aug_cfg.rotate.prob,
            rotation_angle_range=(aug_cfg.rotate.min_angle_deg, aug_cfg.rotate.max_angle_deg),
            zoom_prob=aug_cfg.zoom.prob,
            zoom_in_weight=aug_cfg.zoom.zoom_in.weight,
            zoom_out_weight=aug_cfg.zoom.zoom_out.weight,
            zoom_in_range=(aug_cfg.zoom.zoom_in.factor.min, aug_cfg.zoom.zoom_in.factor.max),
            zoom_out_range=(aug_cfg.zoom.zoom_out.factor.min, aug_cfg.zoom.zoom_out.factor.max)
        )
    else:
        random_augmentation = None  # テストやバリデーションではデータ拡張はなし

    # 共通の変換設定
    label_padding = LabelPaddingTransform(padding_shape=(1, 7), padding_value=0.)
    event_padding = EventPadderTransform(desired_hw=target_hw, mode='constant', value=0)
    format_yolox = YOLOXFormatter(mode=mode)

    # 各モードに応じた変換の設定
    if mode == 'train':
        transform_pipeline = [random_augmentation, label_padding, event_padding, format_yolox]
    else:
        transform_pipeline = [label_padding, event_padding, format_yolox]

    # 変換を統合
    transform_ = transforms.Compose([t for t in transform_pipeline if t is not None])

    # データセットマッピングと生成
    dataset_mapping = {
        'gen1': partial(build_genx_dataset, type='gen1'),
        'gen4': partial(build_genx_dataset, type='gen4'),
    }

    if full_config.dataset.name in dataset_mapping:
        dataset = dataset_mapping[full_config.dataset.name](
            dataset_config=dataset_cfg,
            mode=mode,
            transform=transform_)
    else:
        raise ValueError(f"Unknown dataset name: {full_config.dataset.name}")

    return dataset
