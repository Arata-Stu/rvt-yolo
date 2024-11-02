#from RVT 
# modification torch to numpy

# numpy_representations.py
import numpy as np
import math
from typing import Optional, Tuple

class RepresentationBase:
    def construct(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError
    
    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        raise NotImplementedError

class EventFrame(RepresentationBase):
    def __init__(self, img_size = (480, 640)):
        """
        イベントデータをフレーム形式で表現するクラス。
        高さと幅を指定し、イベントのオン・オフをフレームに描画する。
        """
        self.height, self.width = img_size
        self.channels = 3  # RGB チャンネル (イベントの表示用)

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        # 画像表現のため、np.uint8 (0-255) を使用
        return np.dtype('uint8')

    def get_shape(self) -> Tuple[int, int, int]:
        # (高さ, 幅, 3 チャンネル) の形状を返す
        return (self.channels, self.height, self.width)

    def construct(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        イベントデータからフレームを生成する。
        :param x: イベントのx座標 (np.ndarray)
        :param y: イベントのy座標 (np.ndarray)
        :param pol: イベントの極性（0: オフ, 1: オン）(np.ndarray)
        :param time: イベントのタイムスタンプ（この実装では使用しないが、将来的に拡張可能）(np.ndarray)
        :return: 生成されたフレーム (np.ndarray)
        """
        
        # フレームをRGBの3チャンネル画像として初期化（114のグレー色で埋める）
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 114

        # オン・オフイベントのマスクを作成
        off_events = (pol == 0)
        on_events = (pol == 1)

        # オンイベントを赤、オフイベントを青に割り当て
        frame[y[off_events], x[off_events]] = np.array([0, 0, 255], dtype=np.uint8)  # 青チャンネル（オフイベント）
        frame[y[on_events], x[on_events]] = np.array([255, 0, 0], dtype=np.uint8)    # 赤チャンネル（オンイベント）
        transposed_frames = np.transpose(frame, (2, 0, 1)) 
        return transposed_frames

class StackedHistogram(RepresentationBase):
    def __init__(self, bins= 10, img_size = (480, 640), count_cutoff = 10, fastmode = True):
        """
        Initializes the StackedHistogram representation.

        Args:
            bins (int): Number of time bins.
            height (int): Height of the event frame.
            width (int): Width of the event frame.
            count_cutoff (Optional[int]): Maximum count per bin. Defaults to 255 if None.
            fastmode (bool): If True, uses uint8 for faster computation with potential overflow.
        """
        assert bins >= 1, "Number of bins must be at least 1."
        self.bins = bins
        self.height, self.width = img_size
        self.count_cutoff = count_cutoff if count_cutoff is not None else 255
        self.count_cutoff = min(self.count_cutoff, 255)
        self.fastmode = fastmode
        self.channels = 2  # ON and OFF polarities

    def get_numpy_dtype(self) -> np.dtype:
        return np.uint8

    def get_shape(self) -> Tuple[int, int, int]:
        return (2 * self.bins, self.height, self.width)

    def construct(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        # Initialize representation
        dtype = np.uint8 if self.fastmode else np.int16
        representation = np.zeros((self.channels, self.bins, self.height, self.width), dtype=dtype)

        if len(x) == 0:
            # Return zero representation, converted to uint8
            return representation.astype(np.uint8).reshape(-1, self.height, self.width)

        assert len(x) == len(y) == len(pol) == len(time), "Mismatch in event data lengths."
        assert pol.min() >= 0 and pol.max() <= 1, "Polarity values must be 0 or 1."

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # Normalize time to assign to bins
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int, "End time must be greater than or equal to start time."
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_norm = t_norm * bn
        t_idx = np.floor(t_norm).astype(int)
        t_idx = np.clip(t_idx, a_min=0, a_max=bn - 1)

        # Separate by polarity and accumulate counts
        for p in [0, 1]:
            mask = pol == p
            bin_indices = t_idx[mask]
            x_indices = x[mask]
            y_indices = y[mask]

            # Accumulate counts for each bin
            for b, xi, yi in zip(bin_indices, x_indices, y_indices):
                if 0 <= yi < self.height and 0 <= xi < self.width:
                    representation[p, b, yi, xi] += 1

        # Clamp the counts to prevent overflow
        if self.fastmode:
            np.clip(representation, 0, self.count_cutoff, out=representation)
        else:
            np.clip(representation, 0, self.count_cutoff, out=representation)
            representation = representation.astype(np.uint8)

        # Reshape to (2 * bins, height, width)
        representation = representation.reshape(-1, self.height, self.width)
        return representation


def cumsum_channel_np(x: np.ndarray, num_channels: int) -> np.ndarray:
    """
    Performs a cumulative sum across channels in a NumPy array.

    Args:
        x (np.ndarray): Input array with shape (bins, height, width).
        num_channels (int): Number of channels (bins).

    Returns:
        np.ndarray: Array with cumulative sums.
    """
    for i in reversed(range(num_channels)):
        if i < num_channels - 1:
            x[i] += x[i + 1]
    return x


class MixedDensityEventStack(RepresentationBase):
    def __init__(self, bins: int, img_size = (480, 640), count_cutoff: Optional[int] = None):
        """
        Initializes the MixedDensityEventStack representation.

        Args:
            bins (int): Number of time bins.
            height (int): Height of the event frame.
            width (int): Width of the event frame.
            count_cutoff (Optional[int]): Maximum absolute count per bin. If None, no clipping is applied.
        """
        assert bins >= 1, "Number of bins must be at least 1."
        self.bins = bins
        self.height, self.width = img_size
        self.count_cutoff = count_cutoff
        if self.count_cutoff is not None:
            assert isinstance(count_cutoff, int), "count_cutoff must be an integer."
            assert 0 <= self.count_cutoff <= 127, "count_cutoff must be between 0 and 127."

    def get_numpy_dtype(self) -> np.dtype:
        return np.int8

    def get_shape(self) -> Tuple[int, int, int]:
        return (self.bins, self.height, self.width)

    def construct(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        dtype = np.int8
        representation = np.zeros((self.bins, self.height, self.width), dtype=dtype)

        if len(x) == 0:
            return representation

        assert len(x) == len(y) == len(pol) == len(time), "Mismatch in event data lengths."
        assert pol.min() >= 0 and pol.max() <= 1, "Polarity values must be 0 or 1."
        pol = pol * 2 - 1  # Convert to -1 and +1

        bn, ht, wd = self.bins, self.height, self.width

        # Normalize time using a logarithmic scale
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int, "End time must be greater than or equal to start time."
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_norm = np.clip(t_norm, 1e-6, 1 - 1e-6)

        # Compute bin indices based on logarithmic scaling
        bin_float = bn - np.log(t_norm) / math.log(1 / 2)
        bin_float = np.clip(bin_float, 0, bn - 1)
        t_idx = np.floor(bin_float).astype(int)

        # Accumulate polarity values
        for b, xi, yi, p in zip(t_idx, x, y, pol):
            if 0 <= b < self.bins:
                representation[b, yi, xi] += p

        # Perform cumulative sum across channels
        representation = cumsum_channel_np(representation, self.bins)

        # Apply count cutoff if specified
        if self.count_cutoff is not None:
            np.clip(representation, -self.count_cutoff, self.count_cutoff, out=representation)

        return representation
