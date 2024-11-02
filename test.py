import yaml
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_data_module, fetch_model_module

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import datetime
import argparse

def main(ckpt_path):
    # ckpt_pathからトレーニングのベースディレクトリパスを取得
    train_dir = os.path.dirname(ckpt_path)
    
    # 実行時のタイムスタンプを付与して、一意のテストディレクトリ名を生成
    test_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(train_dir, 'test', test_timestamp)
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成

    # configファイルを読み込み
    config_paths = [
        './config/model/yolox/yolox-s.yaml',
        './config/dataset/gen1/gen1-single.yaml',
        './config/experiment/single/train.yaml'
    ]

    # 各 YAML ファイルを読み込んで OmegaConf にマージ
    configs = [OmegaConf.load(path) for path in config_paths]
    merged_conf = OmegaConf.merge(*configs)
    dynamically_modify_train_config(merged_conf)

    # データモジュールとモデルモジュールのインスタンスを作成
    data = fetch_data_module(merged_conf)
    data.setup('test')
    model = fetch_model_module(merged_conf)
    model.setup('test')

    if ckpt_path != "":
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=train_dir,  # トレーニングディレクトリにテスト結果を保存
        name='test',  # テスト用
        version=test_timestamp  # テストのタイムスタンプをバージョン名に使用
    )

    # トレーナーを設定
    trainer = pl.Trainer(
        logger=logger,  # Loggerに対応させる
        callbacks=None,
        accelerator='gpu',
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmarkを使用して高速化
    )

    # モデルのテストを実行
    trainer.test(model, datamodule=data)

if __name__ == '__main__':
    # argparseでコマンドライン引数を取得
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file")
    
    # 引数をパース
    args = parser.parse_args()

    # ckpt_pathをmainに渡す
    main(args.ckpt_path)
