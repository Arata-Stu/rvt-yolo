from train import main as train  # train.py の main 関数をインポート
import yaml


# config_list.yaml を読み込む
with open("./config/loop_train/config_list.yaml", 'r') as file:
    config_list = yaml.safe_load(file)

# 各設定ファイルのパスを取得
model_configs = config_list['model_configs']
exp_configs = config_list['exp_configs']
dataset_configs = config_list['dataset_configs']

# 全ての設定ファイルの組み合わせでループ
for model_config, exp_config, dataset_config in zip(model_configs, exp_configs, dataset_configs):
    # 各組み合わせで train.py の main 関数を呼び出し
    train(model_config, exp_config, dataset_config)
