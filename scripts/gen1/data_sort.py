import os
import shutil

"""
解凍したばかりのgen1, gen4データセットをディレクトリに整理する
"""

def check_directory_structure(base_dir):
    """
    train, test, val ディレクトリが存在するか確認します。
    """
    required_dirs = ['train', 'test', 'val']
    for d in required_dirs:
        dir_path = os.path.join(base_dir, d)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{d} directory not found in {base_dir}")
    print("All required directories (train, test, val) exist.")

def check_file_pairs(directory):
    """
    npy と h5 ファイルのペアが存在し、同数であることを確認します。
    """
    npy_files = [f for f in os.listdir(directory) if f.endswith('_bbox.npy')]
    h5_files = [f for f in os.listdir(directory) if f.endswith('_td.dat.h5')]

    # ベース名（basename）を取得
    npy_base_names = set([f.replace('_bbox.npy', '') for f in npy_files])
    h5_base_names = set([f.replace('_td.dat.h5', '') for f in h5_files])

    # ベース名の不一致を確認
    missing_npy = h5_base_names - npy_base_names
    missing_h5 = npy_base_names - h5_base_names

    if missing_npy:
        print(f"Missing .npy files for: {missing_npy}")
    if missing_h5:
        print(f"Missing .h5 files for: {missing_h5}")

    # 不足しているペアをスキップするために、共通部分だけを返す
    common_base_names = npy_base_names & h5_base_names
    print(f"Processing {len(common_base_names)} matching file pairs in {directory}")
    
    return common_base_names

def process_directory(base_dir, output_base_dir):
    """
    指定されたディレクトリのすべてのペアに対して、ファイルを新しいディレクトリに移動します。
    """
    check_directory_structure(base_dir)

    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(base_dir, split)  # ディレクトリ内のファイルを直接扱う
        npy_base_names = check_file_pairs(split_dir)

        for base_name in npy_base_names:
            npy_file = os.path.join(split_dir, f"{base_name}_bbox.npy")
            h5_file = os.path.join(split_dir, f"{base_name}_td.dat.h5")

            # 新しいディレクトリを出力先ディレクトリに作成
            new_dir = os.path.join(output_base_dir, split, base_name)
            os.makedirs(new_dir, exist_ok=True)

            # npy と h5 ファイルを新しいディレクトリに移動
            shutil.move(npy_file, os.path.join(new_dir, f"{base_name}_bbox.npy"))
            shutil.move(h5_file, os.path.join(new_dir, f"{base_name}_td.dat.h5"))

            print(f"Moved {base_name} files to {new_dir}")

# 実行部分
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script_name.py /path/to/data_dir /path/to/output_dir")
        sys.exit(1)

    # コマンドライン引数でデータディレクトリと出力ディレクトリを指定
    base_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]
    process_directory(base_data_dir, output_data_dir)
