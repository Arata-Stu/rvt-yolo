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

def main(ckpt_path, model_config, exp_config, dataset_config):
    # ckpt_pathからトレーニングのベースディレクトリパスを取得
    train_dir = os.path.dirname(ckpt_path)
    
    save_dir = os.path.join(train_dir, 'test')
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成

    # 各 YAML ファイルを読み込んで OmegaConf にマージ
    configs = [OmegaConf.load(path) for path in [model_config, exp_config, dataset_config]]
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
        save_dir=save_dir,  # ckptの保存ディレクトリに合わせる
        name='',  # nameを空にすることで、サブディレクトリを作成しない
        version='',
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
    parser.add_argument('--model', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset configuration file')
    
    # 引数をパース
    args = parser.parse_args()

    # 各設定ファイルとckpt_pathをmainに渡す
    main(args.ckpt_path, args.model, args.exp, args.dataset)
