import yaml
import argparse
from train_resume import main as resume_train  # train_resume.py の main 関数を直接インポート

# 引数を解析
parser = argparse.ArgumentParser(description="Run resume_train.py with multiple configurations")
parser.add_argument("--config", type=str, required=True, help="Path to the list.yaml file")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
with open(args.config, 'r') as file:
    config_list = yaml.safe_load(file)

# 各設定ファイルのパスを取得
ckpt_paths = config_list['ckpt_paths']
max_epochs = config_list['max_epochs']

# すべての組み合わせをループして resume_train.py の main 関数を直接呼び出し
for ckpt_path, max_epoch in zip(ckpt_paths, max_epochs):
    print(f"Resuming training with checkpoint: {ckpt_path} and max_epochs: {max_epoch}")
    resume_train(resume_ckpt=ckpt_path, max_epochs=max_epoch)
