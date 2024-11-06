import yaml
from test import main as test  # test.py の main 関数をインポート
import os

def load_ckpt_paths(yaml_path="list.yaml"):
    """YAMLファイルからチェックポイントパスのリストを読み込む"""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['ckpt_paths']

def batch_test(yaml_path="list.yaml"):
    # list.yamlからckptパスを読み込む
    ckpt_paths = load_ckpt_paths(yaml_path)
    
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            print(f"Testing with checkpoint: {ckpt_path}")
            test(ckpt_path)
        else:
            print(f"Checkpoint file {ckpt_path} does not exist. Skipping.")

if __name__ == '__main__':
    # 繰り返し処理を開始
    batch_test()
