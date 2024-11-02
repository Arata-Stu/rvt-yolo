import torch
import numpy as np

def custom_collate_fn(batch):
    # バッチ内の最大シーケンス長を取得
    max_seq_len = max(len(sample['events']) for sample in batch)

    # ラベルの次元数を取得（例では5）
    label_dim = batch[0]['labels'][0].shape[1] if batch[0]['labels'] else 0

    # バッチ内の全タイムステップを通じて最大ラベル数を取得
    max_num_labels = 0
    for sample in batch:
        for labels_at_t in sample['labels']:
            if labels_at_t is not None and labels_at_t.size > 0:
                num_labels = labels_at_t.shape[0]
                if num_labels > max_num_labels:
                    max_num_labels = num_labels

    # パディングされたeventsとlabelsを格納するリストを初期化
    batch_events = []
    batch_labels = []
    batch_img_sizes = []
    batch_is_first_sample = []
    batch_is_padded_mask = []
    batch_timestamp = []

    for sample in batch:
        events = sample['events']
        labels = sample['labels']
        img_size = sample['img_size']
        is_first_sample = sample['is_first_sample']
        is_padded_mask = sample['is_padded_mask']
        timestamps = sample['timestamps']

        # eventsのパディング
        padded_events = []
        for t in range(max_seq_len):
            if t < len(events):
                event = events[t]
            else:
                event = np.zeros_like(events[0])
            padded_events.append(event)
        padded_events = np.stack(padded_events)
        batch_events.append(padded_events)

        # labelsのパディング
        padded_labels = []
        for t in range(max_seq_len):
            if t < len(labels):
                label = labels[t]
                num_labels = label.shape[0]
                if num_labels < max_num_labels:
                    pad_size = max_num_labels - num_labels
                    pad_labels = np.zeros((pad_size, label_dim), dtype=label.dtype)
                    label = np.vstack([label, pad_labels])
            else:
                label = np.zeros((max_num_labels, label_dim), dtype=np.float32)
            padded_labels.append(label)
        padded_labels = np.stack(padded_labels)
        batch_labels.append(padded_labels)

        # その他のフィールドをバッチに追加
        batch_img_sizes.append(img_size)
        batch_is_first_sample.append(is_first_sample)
        batch_is_padded_mask.append(is_padded_mask)
        batch_timestamp.append(timestamps)

    # バッチ全体をテンソルに変換
    batch_events = torch.tensor(np.array(batch_events))
    batch_labels = torch.tensor(np.array(batch_labels))
    batch_img_sizes = torch.tensor(batch_img_sizes)
    batch_is_first_sample = torch.tensor(batch_is_first_sample)
    batch_is_padded_mask = torch.tensor(np.array(batch_is_padded_mask))
    batch_timestamp = torch.tensor([sample['timestamps'] for sample in batch], dtype=torch.float32)


    # バッチ出力
    batch_output = {
        'events': batch_events,
        'labels': batch_labels,
        'img_size': batch_img_sizes,
        'is_first_sample': batch_is_first_sample,
        'is_padded_mask': batch_is_padded_mask,
        'timestamps': batch_timestamp 
    }

    return batch_output
