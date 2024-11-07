import yaml
import argparse
import subprocess

# 引数を解析
parser = argparse.ArgumentParser(description="Run resume_train.py with multiple configurations")
parser.add_argument("--config", type=str, required=True, help="Path to the list.yaml file")
parser.add_argument("--resume_train_script", type=str, required=True, help="Path to resume_train.py script")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
with open(args.config, 'r') as file:
    config_list = yaml.safe_load(file)

# 各設定ファイルのパスを取得
ckpt_paths = config_list['ckpt_paths']
max_epochs = config_list['max_epochs']

# すべての組み合わせをループしてresume_train.pyを呼び出し
for ckpt_path, max_epoch in zip(ckpt_paths, max_epochs):
    # `resume_train.py` のコマンドを作成
    command = [
        "python", args.resume_train_script,
        "--resume_ckpt", ckpt_path,
        "--max_epochs", str(max_epoch)
    ]

    # コマンドを出力して実行
    print("Running command:", " ".join(command))
    subprocess.run(command)
