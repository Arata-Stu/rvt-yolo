import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2


def plot_images_yolox(images, boxes_list=None, mode="train", subplot=False, cols=2):
    """
    複数の画像とバウンディングボックスをプロットする関数
    :param images: 画像のリスト。各画像は (C, H, W) の形状を持つべき。
    :param boxes_list: それぞれの画像に対応するバウンディングボックスのリスト。
                       各ボックスは (class_id, cx, cy, w, h) 形式のリストであるべき。
    :param mode: "train", "val", "test" のいずれか。モードによってボックスの形式が変わる。
    :param subplot: Trueの場合、複数の画像を1枚の図にサブプロットとして表示する。
    :param cols: サブプロットの列数。デフォルトは2。
    """
    
    # boxes_list が与えられていない場合は空のリストとして扱う
    if boxes_list is None:
        boxes_list = [None] * len(images)
    
    # 画像とボックスの数が一致しているか確認
    if len(images) != len(boxes_list):
        raise ValueError("画像の数とバウンディングボックスのリストの数が一致していません。")
    
    # サブプロットを使うかどうかで処理を分岐
    if subplot:
        # 行数の計算
        rows = (len(images) + cols - 1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axs = axs.flatten()  # サブプロットを1次元リストに変換

    for i, (image, boxes) in enumerate(zip(images, boxes_list)):
        ch, height, width = image.shape
        
        if ch not in [1, 3]:
            raise ValueError(f"画像 {i} のチャンネル数は1（グレースケール）または3（RGB）で指定してください。")

        if subplot:
            ax = axs[i]
        else:
            plt.figure(figsize=(6, 6))
            fig, ax = plt.subplots(1)

        if ch == 1:
            # グレースケール画像の場合
            ax.imshow(image[0, :, :], cmap='gray')
        else:
            # RGB画像の場合
            ax.imshow(np.transpose(image, (1, 2, 0)))

        # モードに基づいてボックスを描画
        if boxes is not None:
            for b in boxes:
                if mode == "train":
                    # trainモードでは (class_id, cx, cy, w, h) を使用
                    class_id, cx, cy, w, h = b[:5]

                    if w == 0 or h == 0:
                        continue  # 無効なボックスをスキップ

                    # バウンディングボックスの左上座標 (x_min, y_min) を計算
                    x_min = cx - w / 2
                    y_min = cy - h / 2

                elif mode in ["val", "test"]:
                    # val/testモードでは (x, y, w, h, class) を使用
                    x_min, y_min, w, h, class_id = b[:5]

                    if w == 0 or h == 0:
                        continue  # 無効なボックスをスキップ

                # 四角形のパッチを作成
                rect = patches.Rectangle(
                    (x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')

                # バウンディングボックスを描画
                ax.add_patch(rect)

                # クラスIDをテキストで表示
                label = f"ID: {class_id}"
                ax.text(x_min, y_min - 10, label, color='white', backgroundcolor='red', fontsize=8)

        ax.axis('off')  # 軸を非表示にする

        if not subplot:
            plt.show()

    if subplot:
        # 残りの空白サブプロットを非表示にする
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

def save_sequence_as_video(dataset, start_index: int, end_index: int, t_ms: int, output_file: str):
    """
    指定範囲のサンプルを動画として保存する関数。
    
    Args:
        dataset: チェックするデータセット
        start_index (int): 開始インデックス
        end_index (int): 終了インデックス
        t_ms (int): 各フレームの表示時間 (ミリ秒)
        output_file (str): 出力動画ファイル名
    """
    # FPS計算
    fps = 1000 / t_ms
    
    # サンプルデータから画像サイズを取得
    sample = dataset[start_index]
    _, h, w = sample['events'][0].shape  # (ch, h, w)のshapeを取得
    size = (w, h)
    
    # VideoWriterのセットアップ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    for index in range(start_index, end_index):
        sample = dataset[index]
        events = sample['events']
        labels = sample['labels']
        
        # 各時刻のイベントデータを順番にプロットし、フレームとして追加
        for t, (frame, bbox) in enumerate(zip(events, labels)):
            # 画像データをuint8形式にキャストし、(h, w, 3)の形に変換
            img_uint = np.transpose(frame, (1, 2, 0)).astype('uint8').copy()  # .copy() を追加して連続メモリ化
            
            # bboxの描画
            for box in bbox:
                try:
                    x, y, w, h, cls = box
                    if w > 0 and h > 0:  # 有効なボックスのみ描画
                        start_point = (int(x), int(y))
                        end_point = (int(x + w), int(y + h))
                        
                        # 黄色で太さ2の矩形描画
                        cv2.rectangle(img_uint, start_point, end_point, (0, 255, 255), 2)  
                        cv2.putText(img_uint, f"Cls: {int(cls)}", (int(x), int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error drawing rectangle at index {index}, time step {t}, box {box}: {e}")

            # フレームを動画に追加
            video_writer.write(img_uint)
    
    # 動画ファイルを保存
    video_writer.release()
    print(f"動画が保存されました: {output_file}")


def save_dataloader_sequence_as_video(dataloader, t_ms: int, output_file: str):
    """
    DataLoaderから全範囲のサンプルを動画として保存する関数。

    Args:
        dataloader: DataLoaderオブジェクト
        t_ms (int): 各フレームの表示時間 (ミリ秒)
        output_file (str): 出力動画ファイル名
    """
    # FPS計算
    fps = 1000 / t_ms
    
    # 最初のサンプルデータから画像サイズを取得
    data_iter = iter(dataloader)
    sample = next(data_iter)
    _, _, h, w = sample['events'].shape[-4:]  # (batch, sequence_len, ch, h, w)
    size = (w, h)
    
    # VideoWriterのセットアップ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    # DataLoader全体を順に処理
    for batch_idx, sample in enumerate(dataloader):
        events = sample['events']  # (batch, sequence_len, ch, h, w)
        labels = sample['labels']  # (batch, sequence_len, num_obj, 5)

        # 各バッチ内の時系列に沿ってフレームを追加
        for b in range(events.size(0)):
            for t in range(events.size(1)):
                # フレームの取り出しと変換
                frame = events[b, t].numpy().transpose(1, 2, 0).astype('uint8').copy()  # 明示的にコピー
                label_bboxes = labels[b, t].numpy()

                # 各bboxを描画
                for box in label_bboxes:
                    try:
                        x, y, w, h, cls = box
                        if w > 0 and h > 0:  # 有効なボックスのみ描画
                            start_point = (int(x), int(y))
                            end_point = (int(x + w), int(y + h))
                            
                            # 黄色で太さ2の矩形描画
                            cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
                            cv2.putText(frame, f"Cls: {int(cls)}", (int(x), int(y) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error drawing rectangle at batch {batch_idx}, sample {b}, time step {t}, box {box}: {e}")

                # フレームを動画に追加
                video_writer.write(frame)

    # 動画ファイルを保存
    video_writer.release()
    print(f"動画が保存されました: {output_file}")

def save_dataloader_sequence_as_video_with_predictions(dataloader, t_ms: int, output_file: str, model, conf_thre=0.4, nms_thre=0.45, model_type:str = 'dnn'):
    """
    DataLoaderから全範囲のサンプルを動画として保存する関数。
    ラベルと推論結果のバウンディングボックスをフレームに描画します。

    Args:
        dataloader: DataLoaderオブジェクト
        t_ms (int): 各フレームの表示時間 (ミリ秒)
        output_file (str): 出力動画ファイル名
        model: 推論を行うモデル
        conf_thre (float): 信頼度の閾値
        nms_thre (float): NMSの閾値
    """
    import cv2
    import torch
    from models.detection.yolox.utils.boxes import postprocess
    from data.dataset.genx.classes import GEN1_CLASSES
    from data.data_utils.collate import custom_collate_fn

    # FPS計算
    fps = 1000 / t_ms

    # 最初のサンプルデータから画像サイズを取得
    data_iter = iter(dataloader)
    sample = next(data_iter)
    _, _, _, h, w = sample['events'].shape  # (batch, seq_len, ch, h, w)
    size = (w, h)

    # VideoWriterのセットアップ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    # モデルを評価モードに
    model.eval()

    # デバイスの設定
    device = next(model.parameters()).device

    # DataLoaderを元に戻す
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    if model_type == 'rnn':
        prev_states = None
    # DataLoader全体を順に処理
    for batch_idx, sample in enumerate(dataloader):

        events = sample['events'][:, 0]  # seq_len=1のため、最初のフレームだけ取得 (batch=1, ch, h, w)
        labels = sample['labels'][:, 0]  # (batch=1, num_obj, 5)

        # バッチサイズは1なので、0番目の要素を使用
        events = events[0]  # (ch, h, w)
        labels = labels[0]  # (num_obj, 5)

        # フレームの取り出しと変換
        frame = events.cpu().numpy().transpose(1, 2, 0).astype('uint8').copy()  # 明示的にコピー

        # ラベルのbboxを描画
        label_bboxes = labels.numpy()
        for box in label_bboxes:
            try:
                x, y, w_box, h_box, cls = box
                if w_box > 0 and h_box > 0:  # 有効なボックスのみ描画
                    start_point = (int(x), int(y))
                    end_point = (int(x + w_box), int(y + h_box))

                    # 黄色で太さ2の矩形描画 (ラベル)
                    cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
                    cv2.putText(frame, f"Label: {int(cls)}", (int(x), int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing label rectangle at batch {batch_idx}, box {box}: {e}")

        # モデルへの入力形式に変換
        event_tensor = events.unsqueeze(0).to(torch.float32).to(device)  # (1, ch, h, w)

        # 推論を行い、予測結果を取得
        with torch.no_grad():
            if model_type == 'dnn':
                predictions = model(event_tensor)
            elif model_type == 'rnn':
                predictions, states = model(event_tensor, prev_states)
            processed_preds = postprocess(prediction=predictions,
                                          num_classes=len(GEN1_CLASSES),
                                          conf_thre=conf_thre,
                                          nms_thre=nms_thre,
                                          class_agnostic=False)
            prev_states = states

        # 予測のbboxを描画
        if processed_preds[0] is not None:
            pred_bboxes = processed_preds[0].cpu().numpy()
            for pred_box in pred_bboxes:
                try:
                    x1, y1, x2, y2, conf, conf_class, cls = pred_box
                    start_point = (int(x1), int(y1))
                    end_point = (int(x2), int(y2))

                    # 青色で太さ2の矩形描画 (予測)
                    cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
                    cv2.putText(frame, f"Pred: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error drawing prediction rectangle at batch {batch_idx}, box {pred_box}: {e}")

        # フレームを動画に追加
        video_writer.write(frame)

    # 動画ファイルを保存
    video_writer.release()
    print(f"動画が保存されました: {output_file}")
