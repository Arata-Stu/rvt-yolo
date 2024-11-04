import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初期化用
from pytorch_lightning import LightningModule

def convert_to_trt(lightning_model: LightningModule, input_shape: tuple, trt_engine_path: str):
    """
    PyTorch LightningのモデルをTensorRTに変換して保存する関数

    Args:
        lightning_model (LightningModule): 学習済みのPyTorch Lightningモデル
        input_shape (tuple): モデルの入力シェイプ (例: (1, 3, 224, 224))
        trt_engine_path (str): 出力するTensorRTエンジンの保存パス
    """
    # 1. PyTorch Lightning モデルを ONNX にエクスポート
    dummy_input = torch.randn(*input_shape)
    onnx_path = "temp_model.onnx"
    
    # モデルを推論モードに切り替え
    lightning_model.eval()
    
    # ONNX形式でエクスポート
    torch.onnx.export(
        lightning_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    
    # 2. ONNX を TensorRT に変換
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # TensorRTエンジンをビルドする関数
    def build_engine(onnx_file_path, engine_file_path):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1

            # ONNXモデルを読み込んでパース
            with open(onnx_file_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # TensorRTエンジンをビルドして保存
            engine = builder.build_cuda_engine(network)
            if engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    # ONNXファイルをTensorRTエンジンに変換
    build_engine(onnx_path, trt_engine_path)
    print(f"TensorRTエンジンを {trt_engine_path} に保存しました。")
