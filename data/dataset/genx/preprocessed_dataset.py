from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from .sequence_dataset import GenXRandomAccessDataset, GenXSequentialDataset
from typing import List, Tuple

class GenXPreDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 dataset_type: str = 'gen1',
                 mode: str = 'train',  
                 access_mode: str = 'random',
                 ev_representation_name: str = 'event_frame_dt=50_nbins=10',
                 sequence_length: int = 10,
                 downsample_by_factor_2: bool = False,
                 guarantee_labels: bool = False,
                 only_load_end_labels: bool = False,
                 transform = None  
                 ):
        
        assert mode in ['train', 'val', 'test'], "modeは'train', 'val', 'test'のいずれかである必要があります"
        assert access_mode in ['random', 'stream'], "access_modeは'random' または 'stream' のいずれかである必要があります"
        
        self.root_dir = Path(root_dir)
        subdir_path = self.root_dir / mode
        assert subdir_path.is_dir(), f"{mode}ディレクトリが見つかりません: {subdir_path}"

        self.sequence_datasets = []
        sequence_dirs = [p for p in subdir_path.iterdir() if p.is_dir()]

        for sequence_dir in sequence_dirs:
            if access_mode == 'random':
                # ランダムアクセスモード
                dataset = GenXRandomAccessDataset(
                    dataset_type=dataset_type,
                    sequence_dir=sequence_dir,
                    ev_representation_name=ev_representation_name,
                    sequence_length=sequence_length,
                    downsample_by_factor_2=downsample_by_factor_2,
                    only_load_end_labels=only_load_end_labels,
                    transform=transform
                )
            elif access_mode == 'stream':
                # ストリームモード（GenXSequentialDatasetを使用）
                dataset = GenXSequentialDataset(
                    dataset_type=dataset_type,
                    sequence_dir=sequence_dir,
                    ev_representation_name=ev_representation_name,
                    sequence_length=sequence_length,
                    downsample_by_factor_2=downsample_by_factor_2,
                    guarantee_labels=guarantee_labels,
                    transform=transform
                )
            else:
                raise ValueError("Unsupported access mode specified.")
            
            self.sequence_datasets.append(dataset)

        self.combined_dataset = ConcatDataset(self.sequence_datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index: int):
        return self.combined_dataset[index]

# class GenXPreDataset(Dataset):
#     def __init__(self,
#                  root_dir: str,
#                  dataset_type: str = 'gen1',
#                  mode: str = 'train',  
#                  access_mode: str = 'random',
#                  ev_representation_name: str = 'event_frame_dt=50_nbins=10',
#                  sequence_length: int = 10,
#                  downsample_by_factor_2: bool = False,
#                  guarantee_labels: bool = False,
#                  only_load_end_labels: bool = False,
#                  transform = None  
#                  ):
        
#         assert mode in ['train', 'val', 'test'], "modeは'train', 'val', 'test'のいずれかである必要があります"
#         assert access_mode in ['random', 'stream'], "access_modeは'random' または 'stream' のいずれかである必要があります"
        
#         self.sequence_length = sequence_length
#         self.root_dir = Path(root_dir)
#         subdir_path = self.root_dir / mode
#         assert subdir_path.is_dir(), f"{mode}ディレクトリが見つかりません: {subdir_path}"

#         self.sequence_datasets = []
#         sequence_dirs = [p for p in subdir_path.iterdir() if p.is_dir()]

#         for sequence_dir in sequence_dirs:
#             if access_mode == 'random':
#                 # ランダムアクセスモード
#                 dataset = GenXRandomAccessDataset(
#                     dataset_type=dataset_type,
#                     sequence_dir=sequence_dir,
#                     ev_representation_name=ev_representation_name,
#                     sequence_length=sequence_length,
#                     downsample_by_factor_2=downsample_by_factor_2,
#                     only_load_end_labels=only_load_end_labels,
#                     transform=transform
#                 )
#             elif access_mode == 'stream':
#                 # ストリームモード（GenXSequentialDatasetを使用）
#                 dataset = GenXSequentialDataset(
#                     dataset_type=dataset_type,
#                     sequence_dir=sequence_dir,
#                     ev_representation_name=ev_representation_name,
#                     sequence_length=sequence_length,
#                     downsample_by_factor_2=downsample_by_factor_2,
#                     guarantee_labels=guarantee_labels,
#                     transform=transform
#                 )
#             else:
#                 raise ValueError("Unsupported access mode specified.")
            
#             self.sequence_datasets.append(dataset)

#         # Interleave sequences from different sequence datasets
#         self.all_sequences: List[Tuple[Dataset, int]] = []
#         max_len = max(len(seq_dataset) for seq_dataset in self.sequence_datasets)

#         for idx in range(max_len):
#             for seq_dataset in self.sequence_datasets:
#                 if idx < len(seq_dataset):
#                     self.all_sequences.append((seq_dataset, idx))

#     def __len__(self):
#         return len(self.all_sequences)

#     def __getitem__(self, index):
        
#         seq_id, idx = self.all_sequences[index]

#         # シーケンスデータセットからデータを取得
#         seq_dataset = self.sequence_datasets[seq_id]
#         sample = seq_dataset[idx]
        
#         # 辞書型で返す
#         return {
#             'events': sample['events'],
#             'labels': sample['labels'],
#             'img_size': sample['img_size'],
#             'timestamps': sample['timestamps'],
#             'is_first_sample': sample['is_first_sample'],
#             'is_padded_mask': sample['is_padded_mask']
#         }

