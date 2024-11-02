from omegaconf import DictConfig
from .preprocessed_dataset import GenXPreDataset

def build_genx_dataset(dataset_config: DictConfig, type='gen1', mode="train", transform=None):
    ev_representation = dataset_config.ev_representation
    ev_delta_t = dataset_config.ev_delta_t
    n_bins = dataset_config.n_bins
    ev_representation_name = f'{ev_representation}_dt={ev_delta_t}_nbins={n_bins}'

    # モードに応じた設定を読み込む
    if mode == 'train':
        mode_config = dataset_config.train
    elif mode == 'val':
        mode_config = dataset_config.val
    elif mode == 'test':
        mode_config = dataset_config.test
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    access_mode = mode_config.access_mode
    sequence_length = mode_config.sequence_length
    downsample_by_factor_2 = mode_config.downsample_by_factor_2
    guarantee_labels = mode_config.guarantee_labels
    only_load_end_labels = mode_config.only_load_end_labels

    return GenXPreDataset(root_dir=dataset_config.data_dir,
                          dataset_type=type,
                          mode=mode,
                          access_mode=access_mode,
                          ev_representation_name=ev_representation_name,
                          sequence_length=sequence_length,
                          downsample_by_factor_2=downsample_by_factor_2,
                          guarantee_labels=guarantee_labels,
                          only_load_end_labels=only_load_end_labels,
                          transform=transform)