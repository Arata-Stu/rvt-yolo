import torch.nn as nn
from omegaconf import DictConfig

from ..yolox.yolox import build_darknet
from ..RVT.models.layers.rnn import DWSConvLSTM2d

def build_lstm(lstm_config: DictConfig, inputs_dim):
    return DWSConvLSTM2d(dim=inputs_dim,
                         dws_conv=lstm_config.dws_conv,
                         dws_conv_only_hidden=lstm_config.dws_conv_only_hidden,
                         dws_conv_kernel_size=lstm_config.dws_conv_kernel_size,
                         cell_update_dropout=lstm_config.drop_cell_update)

class Darknet_LSTM(nn.Module):

    def __init__(self, backbone_config: DictConfig):
        super().__init__()

        darknet_cfg = backbone_config.darknet
        lstm_cfg = backbone_config.lstm
        

        # Darknetの構築
        self.darknet = build_darknet(backbone_config=darknet_cfg)
        self.output_keys = darknet_cfg.out_features  # ['dark3', 'dark4', 'dark5']のようなリストが含まれていることを想定
        in_channels = self.darknet.get_stage_dims(self.output_keys)  # (64, 128, 256)のようなタプル

        # 各出力層に対応するLSTMを動的に構築
        self.lstms = nn.ModuleDict({
            key: build_lstm(lstm_cfg, inputs_dim=dim) for key, dim in zip(self.output_keys, in_channels)
        })
        
        self.num_stages = len(self.output_keys)

    def forward(self, x, prev_states=None, token_mask = None):
        # Darknetから特徴マップを取得（辞書形式で["dark3", "dark4", "dark5"]に対応する出力を含む）
        darknet_features = self.darknet(x)
        # 前の状態が渡されなければ、Noneのリストを初期化
        if prev_states is None:
            prev_states = {key: None for key in self.output_keys}
        
        states = {}
        output = {}
        for key in self.output_keys:
            # 各キーに対応する特徴マップとLSTMを使用
            lstm_input = darknet_features[key]  # 特徴マップ
            h_c_tuple = self.lstms[key](lstm_input, prev_states[key])  # 各LSTMで処理
            x = h_c_tuple[0]
            states[key] = h_c_tuple  # 次の時刻ステップ用に状態を保持
            output[key] = x      # 出力辞書に格納

        return output, states

    def get_stage_dims(self, stages):
        return tuple(self.darknet.out_dims[stage_key][0] for stage_key in stages)

    def get_strides(self, stages):
        """
        指定されたステージの出力解像度の縮小倍率を返す。
        
        Parameters:
        stages (str, list, tuple): ステージ名またはステージ名のリスト/タプル
        
        Returns:
        list: 各ステージに対する縮小倍率のリスト
        """
        
        strides = []
        for stage in stages:
            assert stage in self.darknet.out_dims, f"{stage} is not a valid stage."
            stride = int(self.darknet.out_dims[stage][1].split('/')[-1])  # h, w共に同じstride
            strides.append(stride)
        return tuple(strides)


