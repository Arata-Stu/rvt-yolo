import numpy as np
import h5py
import hdf5plugin
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GenXRandomAccessDataset(Dataset):
    def __init__(self,
                 dataset_type,  # 'gen1' or 'gen4'
                 sequence_dir: str,
                 ev_representation_name='event_frame_dt=50_nbins=10',
                 sequence_length=10,
                 downsample_by_factor_2: bool = False,
                 only_load_end_labels: bool = False,
                 transform=None):
        
        self.size = (240, 304) if dataset_type == 'gen1' else (720, 1280)
        self.path = Path(sequence_dir)
        assert self.path.is_dir(), f"指定されたパスが存在しないかディレクトリではありません: {sequence_dir}"
        assert sequence_length >= 1, "sequence_lengthは1以上である必要があります"
        self.sequence_length = sequence_length
        self.downsample_by_factor_2 = downsample_by_factor_2
        self.only_load_end_labels = only_load_end_labels
        self.transform = transform

        # イベント表現ディレクトリとファイルの取得
        self._prepare_event_representation(ev_representation_name)

        # ラベルファイルのパスを取得
        self._prepare_labels()

        # objframe_idxとrepr_idxのマッピングを準備
        self._prepare_indices(ev_representation_name)

    def _prepare_event_representation(self, ev_representation_name: str):
        """イベント表現のパスと形状を準備します。"""
        self.ev_repr_dir = self.path / 'event_representations_v2' / ev_representation_name
        assert self.ev_repr_dir.is_dir(), f'イベント表現ディレクトリが見つかりません: {self.ev_repr_dir}'
        ds_factor_str = '_ds2_nearest' if self.downsample_by_factor_2 else ''
        self.ev_repr_file = self.ev_repr_dir / f'event_representations{ds_factor_str}.h5'
        assert self.ev_repr_file.exists(), f'イベント表現ファイルが見つかりません: {self.ev_repr_file}'

        # イベント表現の数を取得
        with h5py.File(self.ev_repr_file, 'r') as h5f:
            self.num_ev_repr = h5f['data'].shape[0]

    def _prepare_labels(self):
        """ラベルファイルのパスを準備します。"""
        labels_dir = self.path / 'labels_v2'
        assert labels_dir.is_dir(), f'ラベルディレクトリが見つかりません: {labels_dir}'
        self.labels_file = labels_dir / 'labels.npz'
        assert self.labels_file.exists(), f'ラベルファイルが見つかりません: {self.labels_file}'

    def _prepare_indices(self, ev_representation_name: str):
        """インデックスマッピングを準備し、データセットの長さを計算します。"""
        # objframe_idxとrepr_idxのマッピングの読み込み
        objframe_idx_file = self.ev_repr_dir / 'objframe_idx_2_repr_idx.npy'
        assert objframe_idx_file.exists(), f'ファイルが見つかりません: {objframe_idx_file}'
        self.objframe_idx_2_repr_idx = np.load(objframe_idx_file)

        # repr_idxからobjframe_idxへのマッピングを作成
        self.repr_idx_2_objframe_idx = {repr_idx: objframe_idx for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx)}

        # データセットの長さを計算
        self.start_idx_offset = None
        for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx):
            if repr_idx - self.sequence_length + 1 >= 0:
                self.start_idx_offset = objframe_idx
                break
        if self.start_idx_offset is None:
            self.start_idx_offset = len(self.objframe_idx_2_repr_idx)
        self.length = len(self.objframe_idx_2_repr_idx) - self.start_idx_offset

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        corrected_idx = index + self.start_idx_offset
        labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]

        end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.sequence_length
        assert start_idx >= 0, f'start_idx {start_idx} が負の値になっています。'

        # イベント表現の取得
        ev_repr = self.get_event_repr(start_idx, end_idx)

        # ラベルの取得とダウンサンプリング
        labels_list = []
        for repr_idx in range(start_idx, end_idx):
            if self.only_load_end_labels and repr_idx < end_idx - 1:
                labels_list.append(None)
            else:
                # ラベルを取得し、ダウンサンプリングを適用
                label = self.get_labels_from_repr_idx(repr_idx)
                if label is not None and self.downsample_by_factor_2:
                    scaled_label = label.copy()
                    # x, y, w, h のみ1/2にスケール
                    scaled_label['x'] /= 2
                    scaled_label['y'] /= 2
                    scaled_label['w'] /= 2
                    scaled_label['h'] /= 2
                    labels_list.append(scaled_label)
                else:
                    labels_list.append(label)

        # is_padded_maskの作成（パディングがないため全てFalse）
        is_padded_mask = np.array([False] * len(ev_repr))

        # is_first_sampleフラグ
        is_first_sample = True  # ランダムアクセスでは常にTrue

        # 出力の作成
        outputs = {
            'events': ev_repr,
            'labels': labels_list,  # ラベルのリスト（各要素はnumpy.ndarrayまたはNone）
            'img_size': self.size,
            'is_first_sample': is_first_sample,
            'is_padded_mask': is_padded_mask,
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs

    def get_event_repr(self, start_idx: int, end_idx: int):
        """指定された範囲のイベント表現を取得します。"""
        with h5py.File(self.ev_repr_file, 'r') as h5f:
            ev_repr_data = h5f['data'][start_idx:end_idx]
            # 必要なデータのみをnumpy配列として取得
            ev_repr_list = [frame for frame in ev_repr_data]
        return ev_repr_list

    def get_labels_from_repr_idx(self, repr_idx: int):
        """指定されたrepr_idxに対応するラベルを取得します。"""
        objframe_idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        if objframe_idx is None:
            return None
        else:
            # ラベルファイルから必要なラベルをオンデマンドで読み込む
            label_data = np.load(self.labels_file, allow_pickle=True)
            labels = label_data['labels']
            objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']

            from_idx = objframe_idx_2_label_idx[objframe_idx]
            if objframe_idx + 1 < len(objframe_idx_2_label_idx):
                to_idx = objframe_idx_2_label_idx[objframe_idx + 1]
            else:
                to_idx = len(labels)
            labels_for_frame = labels[from_idx:to_idx]

            if labels_for_frame.size > 0:
                return labels_for_frame
            else:
                return None
        
class GenXSequentialDataset(Dataset):
    def __init__(
        self,
        dataset_type,  # 'gen1' or 'gen4'
        sequence_dir: str,
        ev_representation_name='event_frame_dt=50_nbins=10',
        sequence_length=10,
        downsample_by_factor_2: bool = False,
        guarantee_labels: bool = False,
        transform=None):
        """
        Initialize the SequenceDataset.

        Args:
            sequence_dir (Path): Path to the sequence directory.
            ev_representation_name (str): Name of the event representation.
            sequence_length (int): Desired sequence length.
            dataset_type (str): Type of the dataset ('gen1' or 'gen4').
            downsample_by_factor_2 (bool): Whether to downsample by a factor of 2.
            guarantee_labels (bool): Ensure each sample contains labels.
        """
        super().__init__()
        self.size = (240, 304) if dataset_type == 'gen1' else (720, 1280)
        self.sequence_dir = Path(sequence_dir)
        self.ev_representation_name = ev_representation_name
        self.sequence_length = sequence_length
        self.dataset_type = dataset_type
        self.downsample_by_factor_2 = downsample_by_factor_2
        self.guarantee_labels = guarantee_labels
        self.transform = transform

        # Prepare paths and minimal data
        self._prepare_paths_and_indices()

    def _prepare_paths_and_indices(self):
        """Prepare paths to event representations and labels, and minimal indices."""

        # Prepare event representations path
        ds_factor_str = '_ds2_nearest' if self.downsample_by_factor_2 else ''
        ev_repr_dir = (
            self.sequence_dir / 'event_representations_v2' / self.ev_representation_name
        )
        ev_repr_file = ev_repr_dir / f'event_representations{ds_factor_str}.h5'
        if not ev_repr_file.exists():
            raise FileNotFoundError(f"Event representations not found at {ev_repr_file}")
        self.ev_repr_file = ev_repr_file  # Store path to HDF5 file

        # Open HDF5 file to get shape without loading data
        with h5py.File(str(ev_repr_file), 'r') as h5f:
            self.ev_repr_shape = h5f['data'].shape  # (num_frames, H, W)

        # Load mapping from object frame indices to representation indices
        objframe_idx_file = ev_repr_dir / 'objframe_idx_2_repr_idx.npy'
        if not objframe_idx_file.exists():
            raise FileNotFoundError(f"Mapping file not found at {objframe_idx_file}")
        self.objframe_idx_2_repr_idx = np.load(str(objframe_idx_file))

        # Create mapping from representation index to object frame index
        self.repr_idx_2_objframe_idx = {
            repr_idx: objframe_idx
            for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx)
        }

        # Prepare labels path
        labels_dir = self.sequence_dir / 'labels_v2'
        labels_file = labels_dir / 'labels.npz'
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels not found at {labels_file}")
        self.labels_file = labels_file  # Store path to labels file

        # Load minimal label data
        label_data = np.load(str(labels_file))
        # We only load the indices mapping here
        self.objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']

        # Prepare indices for sequence samples
        self._prepare_indices()

    def _prepare_indices(self):
        """Prepare start and end indices for sequences."""
        num_ev_repr = self.ev_repr_shape[0]

        if self.guarantee_labels:
            # Get indices where labels are available
            labeled_repr_indices = [
                repr_idx
                for repr_idx in range(num_ev_repr)
                if self._repr_idx_has_label(repr_idx)
            ]
            # Split sequences to ensure each has labels
            self.start_indices = []
            idx = 0
            while idx < len(labeled_repr_indices):
                start_idx = max(
                    labeled_repr_indices[idx] - self.sequence_length + 1, 0
                )
                end_idx = start_idx + self.sequence_length
                if end_idx > num_ev_repr:
                    end_idx = num_ev_repr
                    start_idx = max(end_idx - self.sequence_length, 0)
                self.start_indices.append((start_idx, end_idx))
                idx += 1
        else:
            # Create sequences starting from every possible index
            self.start_indices = []
            for start_idx in range(0, num_ev_repr, self.sequence_length):
                end_idx = min(start_idx + self.sequence_length, num_ev_repr)
                self.start_indices.append((start_idx, end_idx))

        self.length = len(self.start_indices)

    def _repr_idx_has_label(self, repr_idx: int):
        objframe_idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        if objframe_idx is None:
            return False
        else:
            label_indices = self.objframe_idx_2_label_idx[objframe_idx]
            return label_indices.size > 0

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        start_idx, end_idx = self.start_indices[index]
        sample_len = end_idx - start_idx

        # Open the HDF5 file and read the required event representations
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            ev_repr_data = h5f['data'][start_idx:end_idx]
        # Convert to list
        ev_repr_list = [ev_repr_data[i] for i in range(ev_repr_data.shape[0])]

        # Load labels.npz
        label_data = np.load(str(self.labels_file))
        labels = label_data['labels']
        objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']

        labels_list = []
        for idx in range(start_idx, end_idx):
            # Get labels for repr_idx idx
            objframe_idx = self.repr_idx_2_objframe_idx.get(idx, None)
            if objframe_idx is None:
                labels_list.append(None)
            else:
                from_idx = objframe_idx_2_label_idx[objframe_idx]
                to_idx = objframe_idx_2_label_idx[objframe_idx + 1] if objframe_idx + 1 < len(objframe_idx_2_label_idx) else len(labels)
                labels_for_frame = labels[from_idx:to_idx]
                if labels_for_frame.size > 0:
                    if self.downsample_by_factor_2:
                        # x, y, w, h のみを1/2スケールに変換
                        scaled_labels = labels_for_frame.copy()
                        scaled_labels['x'] /= 2
                        scaled_labels['y'] /= 2
                        scaled_labels['w'] /= 2
                        scaled_labels['h'] /= 2
                        labels_list.append(scaled_labels)
                    else:
                        labels_list.append(labels_for_frame)
                else:
                    labels_list.append(None)

        # Padding if necessary
        is_padded_mask = np.array([False] * sample_len)
        if sample_len < self.sequence_length:
            padding_len = self.sequence_length - sample_len
            padding_shape = (padding_len,) + ev_repr_data.shape[1:]
            padding_repr = [np.zeros(ev_repr_data.shape[1:], dtype=ev_repr_data.dtype) for _ in range(padding_len)]
            ev_repr_list.extend(padding_repr)  # パディング部分をリストに追加
            labels_list.extend([None] * padding_len)
            is_padded_mask = np.concatenate(
                (is_padded_mask, np.array([True] * padding_len)), axis=0
            )

        # is_first_sample flag
        is_first_sample = index == 0

        outputs = {
            'events': ev_repr_list,            # リスト形式に変更
            'labels': labels_list,             # List of labels or None
            'img_size': self.size,             # Tuple (H, W)
            'is_first_sample': is_first_sample,  # True if first sample in the sequence
            'is_padded_mask': is_padded_mask,  # Mask indicating padded frames
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs