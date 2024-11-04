import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        
        dataset_cfg = config.dataset

        dataset_name = dataset_cfg.name
        assert dataset_name in {'gen1', 'gen4'}
        dataset_hw = dataset_cfg.orig_size

        mdl_cfg = config.model
        mdl_name = mdl_cfg.name
        if mdl_name == 'rvt':
            backbone_cfg = mdl_cfg.backbone
            backbone_name = backbone_cfg.name
            if backbone_name == 'MaxViTRNN' or backbone_name == 'MaxViTRNN-SSM':
                partition_split_32 = backbone_cfg.partition_split_32
                assert partition_split_32 in (1, 2, 4)

                multiple_of = 32 * partition_split_32
                mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=multiple_of)
                print(f'Set {backbone_name} backbone (height, width) to {mdl_hw}')
                dataset_cfg.target_size = mdl_hw
                backbone_cfg.in_res_hw = mdl_hw

                attention_cfg = backbone_cfg.stage.attention
                partition_size = tuple(x // (32 * partition_split_32) for x in mdl_hw)
                assert (mdl_hw[0] // 32) % partition_size[0] == 0, f'{mdl_hw[0]=}, {partition_size[0]=}'
                assert (mdl_hw[1] // 32) % partition_size[1] == 0, f'{mdl_hw[1]=}, {partition_size[1]=}'
                print(f'Set partition sizes: {partition_size}')
                attention_cfg.partition_size = partition_size
            else:
                print(f'{backbone_name=} not available')
                raise NotImplementedError
            num_classes = 2 if dataset_name == 'gen1' else 3
            mdl_cfg.head.num_classes = num_classes
            print(f'Set {num_classes=} for detection head')
        elif mdl_name == 'yolox' or mdl_name == 'yolox-lstm':
            partition_split_32 = mdl_cfg.partition_split_32
            multiple_of = 32 * partition_split_32
            mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=multiple_of)
            mdl_hw = _get_square_hw(hw=mdl_hw)
            dataset_cfg.target_size = mdl_hw

            class_len_map = {
                'gen1': 2,
                'gen4': 3,
                'dsec': 8
            }
            mdl_cfg.head.num_classes = class_len_map[dataset_name]
        else:
            print(f'{mdl_name=} not available')
            raise NotImplementedError


def _get_modified_hw_multiple_of(hw: Tuple[int, int], multiple_of: int) -> Tuple[int, ...]:
    assert len(hw) == 2
    assert isinstance(multiple_of, int)
    assert multiple_of >= 1
    if multiple_of == 1:
        return hw
    new_hw = tuple(math.ceil(x / multiple_of) * multiple_of for x in hw)
    return new_hw

def _get_square_hw(hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    高さと幅を指定された倍数の範囲で正方形にする。

    :param hw: 元の高さと幅
    :param multiple_of: サイズを揃える際に使用する倍数
    :return: 正方形の高さと幅 (multiple_of の倍数)
    """
    # 高さと幅の最大値を使用して、正方形のサイズを計算
    max_hw = max(hw)
    return (max_hw, max_hw)