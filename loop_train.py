from train import main as train  # train.py の main 関数をインポート
import yaml
from itertools import product
import argparse

# 引数を解析
parser = argparse.ArgumentParser(description="Configuration file for training loop")
parser.add_argument("--config", type=str, required=True, help="Path to the list.yaml file")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
with open(args.config, 'r') as file:
    config_list = yaml.safe_load(file)

# 各設定ファイルのパスを取得
model_configs = config_list['model_configs']
exp_configs = config_list['exp_configs']
dataset_configs = config_list['dataset_configs']
ckpt_paths = config_list.get('ckpt_paths', [None] * len(dataset_configs))

# すべての組み合わせをループ
for (model_config, exp_config, dataset_config), ckpt_path in product(
        zip(model_configs, exp_configs, dataset_configs), ckpt_paths):
    # 各組み合わせで train.py の main 関数を呼び出し、ckpt_pathが存在する場合はそれを使用して再開
    train(model_config, exp_config, dataset_config, resume_ckpt=ckpt_path)
