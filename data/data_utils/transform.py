import numpy as np
import cv2
import torch 

from .bbox import xywh2cxcywh


class EventPadderTransform:
    def __init__(self, desired_hw, mode='constant', value=0):
        """
        :param desired_hw: Desired height and width for the events.
        :param mode: Padding mode (e.g., 'constant', 'reflect', 'replicate').
        :param value: Padding value when mode is 'constant'.
        """
        assert isinstance(desired_hw, tuple), "desired_hw should be a tuple of (height, width)"
        assert len(desired_hw) == 2, "desired_hw should contain exactly two elements (height, width)"
        self.desired_hw = desired_hw
        self.mode = mode
        self.value = value

    @staticmethod
    def _pad_numpy(input_array, desired_hw, mode, value):
        """
        Pads the input array to the desired height and width using numpy.
        :param input_array: NumPy array of shape (num_channels, height, width).
        :param desired_hw: Target height and width.
        :param mode: Padding mode.
        :param value: Padding value if mode is 'constant'.
        :return: Padded NumPy array.
        """
        ht, wd = input_array.shape[-2:]  # Get the current height and width of the input array.
        ht_des, wd_des = desired_hw  # Get the desired height and width.

        # Ensure the desired dimensions are greater than or equal to the current dimensions.
        assert ht <= ht_des, f"Current height {ht} exceeds desired height {ht_des}"
        assert wd <= wd_des, f"Current width {wd} exceeds desired width {wd_des}"

        # Calculate padding amounts
        pad_top = 0
        pad_bottom = ht_des - ht
        pad_left = 0
        pad_right = wd_des - wd

        # Apply padding using np.pad
        pad_width = [(0, 0),  # No padding for the channels
                     (pad_top, pad_bottom),  # Padding for height
                     (pad_left, pad_right)]  # Padding for width

        padded_array = np.pad(input_array, pad_width, mode=mode, constant_values=value)
        
        return padded_array

    def __call__(self, inputs):
        """
        Apply padding to each 'events' numpy array in the inputs.
        :param inputs: Dictionary containing 'events', a list of numpy arrays of shape (num_channels, height, width).
        :return: Dictionary with each 'events' numpy array padded.
        """
        events_list = inputs['events']
        
        # Ensure the input is a list of numpy arrays
        assert isinstance(events_list, list), "'events' should be a list of numpy arrays"
        assert all(isinstance(events, np.ndarray) and len(events.shape) == 3 for events in events_list), \
            "Each element in 'events' should be a numpy array with shape (num_channels, height, width)"
        
        # Apply padding to each events array in the list
        padded_events_list = [
            self._pad_numpy(events, desired_hw=self.desired_hw, mode=self.mode, value=self.value)
            for events in events_list
        ]

        
        outputs = {
            'events': padded_events_list,                  # 元のイベントフレーム
            'labels': inputs['labels'],                           # ラベルリスト（Noneは0でパディング済み）
            'img_size': inputs['img_size'],                  # オリジナルの画像サイズ
            'is_first_sample': inputs['is_first_sample'],    # ランダム読み込みで常にTrue
            'is_padded_mask': inputs['is_padded_mask']       # パディングマスク
        }
        return outputs
    
class LabelPaddingTransform:
    def __init__(self, padding_shape=(1,), padding_value=0.):
        """
        ラベルのパディング設定
        :param padding_shape: パディングするラベルの形状（デフォルトは(1,)）
        :param padding_value: パディングする際の値（デフォルトは0.0）
        """
        self.padding_shape = padding_shape
        self.padding_value = padding_value

        # dtype を定義
        self.dtype = np.dtype([
            ('t', '<u8'), 
            ('x', '<f4'), 
            ('y', '<f4'), 
            ('w', '<f4'), 
            ('h', '<f4'), 
            ('class_id', 'u1'), 
            ('class_confidence', '<f4'), 
            ('track_id', '<u4')
        ])

        # 単一のゼロ埋めパディングデータ
        self.padding_label = np.array([(0, 0., 0., 0., 0., 0, 0., 0)], dtype=self.dtype)

    def __call__(self, inputs):
        labels = inputs['labels']
        labels_list = []  # ラベルリストを初期化

        # ラベルリストをループ処理
        for label in labels:
            if label is None:
                # Noneの場合、単一のゼロ埋めデータを追加
                labels_list.append(self.padding_label)
            else:
                # Noneでない場合はそのままリストに追加
                labels_list.append(label)

        outputs = {
            'events': inputs['events'],                      # 元のイベントフレーム
            'labels': labels_list,                           # ラベルリスト（Noneは0でパディング済み）
            'img_size': inputs['img_size'],                  # オリジナルの画像サイズ
            'is_first_sample': inputs['is_first_sample'],    # ランダム読み込みで常にTrue
            'is_padded_mask': inputs['is_padded_mask']       # パディングマスク
        }
        return outputs
    
class ResizeTransform:
    def __init__(self, target_size):
        """
        :param target_size: リサイズ後の目標サイズ (width, height)
        """
        assert isinstance(target_size, tuple) and len(target_size) == 2, \
            "target_size should be a tuple of (width, height)"
        self.target_size = target_size

    @staticmethod
    def _resize_with_aspect_ratio(input_array, target_size):
        """
        アスペクト比を維持しながら、指定のサイズにリサイズします。
        :param input_array: リサイズする画像データ (num_channels, height, width)
        :param target_size: 目標サイズ (width, height)
        :return: リサイズされた画像データ
        """
        num_channels, ht, wd = input_array.shape
        target_w, target_h = target_size

        # 各次元に対する比率を計算し、アスペクト比を維持するために小さい方を選択
        ratio_w = target_w / wd
        ratio_h = target_h / ht
        scale_ratio = min(ratio_w, ratio_h)

        # 新しいサイズを計算
        new_w = int(wd * scale_ratio)
        new_h = int(ht * scale_ratio)

        # 各チャネルを個別にリサイズし、アスペクト比を維持
        resized_channels = [
            cv2.resize(input_array[channel], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            for channel in range(num_channels)
        ]
        
        # 結果をstackして新しい形状にする
        resized_array = np.stack(resized_channels, axis=0)
        return resized_array

    def __call__(self, inputs):
        """
        'events'内の各numpy配列をリサイズします。
        :param inputs: 'events'を含む辞書。 'events'はshape (num_channels, height, width) のnumpy配列のリスト。
        :return: リサイズされた'events'を持つ辞書。
        """
        events_list = inputs['events']
        
        # リサイズを各イベントデータに適用
        resized_events_list = [
            self._resize_with_aspect_ratio(events, target_size=self.target_size)
            for events in events_list
        ]
        
        outputs = {
            'events': resized_events_list,                   # リサイズ済みのイベントフレーム
            'labels': inputs['labels'],                      # 元のラベルリスト
            'img_size': inputs['img_size'],                  # オリジナルの画像サイズ
            'is_first_sample': inputs['is_first_sample'],    # ランダム読み込みで常にTrue
            'is_padded_mask': inputs['is_padded_mask']       # パディングマスク
        }
        return outputs

class YOLOXFormatter:
    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, inputs):
        # ラベルとタイムスタンプのリストを初期化
        labels_list = []
        timestamps_list = []

        # ラベルリストをループ処理
        for label in inputs['labels']:
            if label is None:
                # Noneの場合は全て0のラベルデータを追加し、タイムスタンプも0を追加
                labels_list.append(np.zeros((1, 5), dtype=np.float32))
                timestamps_list.append(0.0)  # タイムスタンプの代替値（0などの適切な値を設定）
            else:
                # ラベル情報を取得
                x, y, w, h, class_id, t = label['x'], label['y'], label['w'], label['h'], label['class_id'], label['t']
                
                # tが配列やリスト形式かどうかを確認
                if isinstance(t, (list, np.ndarray)):
                    t = float(t[0])  # 最初の要素を使用
                else:
                    t = float(t)
                # モードに応じてラベル形式を選択
                if self.mode == 'train':
                    # 中心座標(cx, cy)形式に変換
                    bboxes = np.stack((x, y, w, h), axis=-1)
                    bboxes_cxcywh = xywh2cxcywh(bboxes.copy())
                    target = np.hstack((class_id.reshape(-1, 1), bboxes_cxcywh))
                else:
                    # バウンディングボックスをそのまま使用
                    bboxes_xywh = np.stack((x, y, w, h), axis=-1)
                    target = np.hstack((bboxes_xywh, class_id.reshape(-1, 1)))

                labels_list.append(target)
                timestamps_list.append(float(t))  # 単一のタイムスタンプ値として追加

        # 出力の辞書を準備
        outputs = {
            'events': inputs['events'],                     # 元のイベントフレーム
            'labels': labels_list,                          # ラベルリスト（Noneは0でパディング済み）
            'timestamps': timestamps_list,                  # タイムスタンプリスト
            'img_size': inputs['img_size'],                 # オリジナルの画像サイズ
            'is_first_sample': inputs['is_first_sample'],   # ランダム読み込みで常にTrue
            'is_padded_mask': inputs['is_padded_mask']      # パディングマスク
        }
        
        return outputs

def flip_horizontal(image, labels):
    """
    Flip the image and labels horizontally.

    Args:
        image (numpy.ndarray): Image array of shape (D, H, W).
        labels (numpy.ndarray or None): Labels array with fields 'x', 'y', 'w', 'h', or None.

    Returns:
        flipped_image (numpy.ndarray): Horizontally flipped image.
        flipped_labels (numpy.ndarray or None): Labels adjusted for the flipped image, or None.
    """

    # Flip the image horizontally
    flipped_image = image[:, :, ::-1]
    image_width = image.shape[2]

    if labels is not None:
        # Adjust the labels
        labels = labels.copy()
        # Keep labels with padding values (x=0, w=0) unchanged
        mask = (labels['x'] != 0) | (labels['w'] != 0)
        labels['x'][mask] = image_width - labels['x'][mask] - labels['w'][mask]
    else:
        labels = None

    return flipped_image, labels

def rotate(image, labels, angle):
    """
    Rotate the image and labels by a given angle.

    Args:
        image (numpy.ndarray): Image array of shape (D, H, W).
        labels (numpy.ndarray or None): Labels array with fields 'x', 'y', 'w', 'h', or None.
        angle (float): Rotation angle in degrees. Positive values mean counter-clockwise rotation.

    Returns:
        rotated_image (numpy.ndarray): Rotated image.
        rotated_labels (numpy.ndarray or None): Labels adjusted for the rotated image, or None.
    """
    D, H, W = image.shape
    center = (W / 2, H / 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Prepare image for cv2 (transpose to (H, W, D))
    image_cv2 = np.transpose(image, (1, 2, 0))

    # Apply rotation
    rotated_image_cv2 = cv2.warpAffine(
        image_cv2, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Convert back to original shape (D, H, W)
    rotated_image = np.transpose(rotated_image_cv2, (2, 0, 1))

    if labels is not None:
        # Rotate the labels
        labels = labels.copy()
        new_boxes = []

        # Compute rotation matrix for labels
        theta = np.deg2rad(-angle)  # Note: cv2 uses opposite sign for angle
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        for label in labels:
            x, y, w, h = label['x'], label['y'], label['w'], label['h']
            # Define the four corners of the bounding box
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ])

            # Shift corners to center
            corners_centered = corners - center

            # Apply rotation
            rotated_corners = np.dot(corners_centered, rotation_matrix.T)

            # Shift back
            rotated_corners += center

            # Get new bounding box
            x_coords = rotated_corners[:, 0]
            y_coords = rotated_corners[:, 1]
            x_min = x_coords.min()
            y_min = y_coords.min()
            x_max = x_coords.max()
            y_max = y_coords.max()

            new_w = x_max - x_min
            new_h = y_max - y_min

            new_boxes.append((x_min, y_min, new_w, new_h))

        # Update labels
        labels['x'] = np.array([box[0] for box in new_boxes])
        labels['y'] = np.array([box[1] for box in new_boxes])
        labels['w'] = np.array([box[2] for box in new_boxes])
        labels['h'] = np.array([box[3] for box in new_boxes])
    else:
        labels = None

    return rotated_image, labels

def clip_bboxes(labels, image_shape):
    """
    Clip bounding boxes to ensure they remain within the image bounds.

    Args:
        labels (numpy.ndarray or None): Array with fields 'x', 'y', 'w', 'h', or None.
        image_shape (Tuple[int, int]): Shape of the image (height, width).

    Returns:
        clipped_labels (numpy.ndarray or None): Adjusted bounding boxes within image bounds, or None.
    """
    if labels is None:
        return None

    labels = labels.copy()
    height, width = image_shape

    # Clip x, y coordinates to be within the image bounds
    labels['x'] = np.clip(labels['x'], 0, width)
    labels['y'] = np.clip(labels['y'], 0, height)

    # Clip width and height so that the bounding boxes don't extend beyond the image
    labels['w'] = np.clip(labels['w'], 0, width - labels['x'])
    labels['h'] = np.clip(labels['h'], 0, height - labels['y'])

    return labels

def remove_flat_labels(labels):
    """
    Remove flat labels (where w <= 0 or h <= 0) from the labels array.

    Args:
        labels (numpy.ndarray or None): Labels array with fields 'x', 'y', 'w', 'h', or None.

    Returns:
        filtered_labels (numpy.ndarray or None): Labels with w > 0 and h > 0, or None if no labels remain.
    """
    if labels is None:
        return None

    # Keep only labels that satisfy the condition
    mask = (labels['w'] > 0) & (labels['h'] > 0)
    filtered_labels = labels[mask]

    # Return None if no labels remain
    if len(filtered_labels) == 0:
        return None

    return filtered_labels

def find_zoom_center(labels_list):
    """
    Find the center for zoom based on labels from all frames.

    Args:
        labels_list (list): List of labels for each frame.

    Returns:
        Tuple[int, int] or None: Coordinates for zoom center (x, y), or None if no valid labels.
    """
    possible_centers = []

    for labels in labels_list:
        if labels is not None and len(labels) > 0:
            for label in labels:
                center_x = label['x'] + label['w'] / 2
                center_y = label['y'] + label['h'] / 2
                possible_centers.append((center_x, center_y))

    if possible_centers:
        avg_center_x = int(np.mean([c[0] for c in possible_centers]))
        avg_center_y = int(np.mean([c[1] for c in possible_centers]))
        return avg_center_x, avg_center_y

    return None

def zoom_in(image, labels, zoom_factor, center=None):
    """
    Zoom in on the image and adjust labels accordingly.

    Args:
        image (numpy.ndarray): Image array of shape (D, H, W).
        labels (numpy.ndarray or None): Labels array with fields 'x', 'y', 'w', 'h', or None.
        zoom_factor (float): Factor by which to zoom in (>1).
        center (Tuple[int, int], optional): Coordinates (x, y) for the center of zoom.

    Returns:
        zoomed_image (numpy.ndarray): Zoomed-in image.
        zoomed_labels (numpy.ndarray or None): Labels adjusted for the zoomed image, or None.
    """
    D, H, W = image.shape
    new_H = int(H / zoom_factor)
    new_W = int(W / zoom_factor)

    # Zoom window position: use center if provided, else random
    if center is None:
        x1 = np.random.randint(0, W - new_W + 1)
        y1 = np.random.randint(0, H - new_H + 1)
    else:
        cx, cy = center
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))

    # Crop the image
    cropped_image = image[:, y1:y1 + new_H, x1:x1 + new_W]

    # Resize back to original size using cv2
    cropped_image_cv2 = np.transpose(cropped_image, (1, 2, 0))  # (H, W, D)
    zoomed_image_cv2 = cv2.resize(cropped_image_cv2, (W, H), interpolation=cv2.INTER_CUBIC)
    zoomed_image = np.transpose(zoomed_image_cv2, (2, 0, 1))  # (D, H, W)

    if labels is not None:
        # Adjust labels
        labels = labels.copy()
        labels['x'] = (labels['x'] - x1) * zoom_factor
        labels['y'] = (labels['y'] - y1) * zoom_factor
        labels['w'] = labels['w'] * zoom_factor
        labels['h'] = labels['h'] * zoom_factor

        # Clip the bounding boxes to ensure they remain within the image bounds
        zoomed_labels = clip_bboxes(labels, (H, W))
        zoomed_labels = remove_flat_labels(zoomed_labels)  # Apply condition w > 0 and h > 0
    else:
        zoomed_labels = None

    return zoomed_image, zoomed_labels

def zoom_out(image, labels, zoom_factor, center=None):
    """
    Zoom out the image by scaling down and placing it in a canvas.

    Args:
        image (numpy.ndarray): Image array of shape (D, H, W).
        labels (numpy.ndarray or None): Labels array with fields 'x', 'y', 'w', 'h', or None.
        zoom_factor (float): Factor by which to zoom out (>1).
        center (Tuple[int, int], optional): Coordinates (x, y) for the center of zoom.

    Returns:
        zoomed_image (numpy.ndarray): Zoomed-out image with padding.
        zoomed_labels (numpy.ndarray or None): Labels adjusted for the zoomed image, or None.
    """
    D, H, W = image.shape
    new_H = int(H / zoom_factor)
    new_W = int(W / zoom_factor)

    # Resize the image using cv2
    image_cv2 = np.transpose(image, (1, 2, 0))  # (H, W, D)
    resized_image_cv2 = cv2.resize(image_cv2, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
    resized_image = np.transpose(resized_image_cv2, (2, 0, 1))  # (D, new_H, new_W)

    # Create a canvas and place the resized image at the specified or random position
    canvas = np.zeros_like(image)
    if center is None:
        x1 = np.random.randint(0, W - new_W + 1)
        y1 = np.random.randint(0, H - new_H + 1)
    else:
        cx, cy = center
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))
            
    canvas[:, y1:y1 + new_H, x1:x1 + new_W] = resized_image

    if labels is not None:
        # Adjust labels
        labels = labels.copy()
        labels['x'] = labels['x'] / zoom_factor + x1
        labels['y'] = labels['y'] / zoom_factor + y1
        labels['w'] = labels['w'] / zoom_factor
        labels['h'] = labels['h'] / zoom_factor

        # Clip the bounding boxes to ensure they remain within the image bounds
        zoomed_labels = clip_bboxes(labels, (H, W))
        zoomed_labels = remove_flat_labels(zoomed_labels)  # Apply condition w > 0 and h > 0
    else:
        zoomed_labels = None

    return canvas, zoomed_labels

class RandomSpatialAugmentor:
    def __init__(self,
                 h_flip_prob=0.5,
                 rotation_prob=0.5,
                 rotation_angle_range=(-6, 6),
                 zoom_in_weight=8,
                 zoom_out_weight=2,
                 zoom_in_range=(1.0, 1.5),
                 zoom_out_range=(1.0, 1.5),
                 zoom_prob=0.0):
        self.h_flip_prob = h_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angle_range = rotation_angle_range
        self.zoom_in_weight = zoom_in_weight
        self.zoom_out_weight = zoom_out_weight
        self.zoom_in_range = zoom_in_range
        self.zoom_out_range = zoom_out_range
        self.zoom_prob = zoom_prob

        # Zoom operation weighted distribution
        self.zoom_in_or_out_distribution = torch.distributions.categorical.Categorical(
            probs=torch.tensor([zoom_in_weight, zoom_out_weight], dtype=torch.float)
        )

    def __call__(self, inputs):
        evframes_list = inputs['events']  # [(ch, h, w) * length]
        labels_list = inputs['labels']  # [labels_array * length]
        
        augmented_evframes_list = []
        augmented_labels_list = []
        
        # Decide random transformation parameters once
        apply_h_flip = np.random.rand() < self.h_flip_prob
        apply_rotation = np.random.rand() < self.rotation_prob
        angle = np.random.uniform(*self.rotation_angle_range) if apply_rotation else None
        apply_zoom = np.random.rand() < self.zoom_prob

        # Zoom settings
        zoom_func = None
        zoom_factor = None
        if apply_zoom:
            zoom_choice = self.zoom_in_or_out_distribution.sample().item()
            if zoom_choice == 0:  # Zoom In
                zoom_factor = np.random.uniform(*self.zoom_in_range)
                zoom_func = zoom_in
            else:  # Zoom Out
                zoom_factor = np.random.uniform(*self.zoom_out_range)
                zoom_func = zoom_out
        
        # Determine zoom center coordinates (common for all frames)
        zoom_center = find_zoom_center(labels_list)
        if zoom_center is None:
            # Use image center by default
            _, H, W = evframes_list[0].shape
            zoom_center = (W // 2, H // 2)
        
        # Apply the same preprocessing to each frame
        for evframes, labels in zip(evframes_list, labels_list):
            # Horizontal Flip
            if apply_h_flip:
                evframes, labels = flip_horizontal(evframes, labels)

            # Rotation
            if apply_rotation:
                evframes, labels = rotate(evframes, labels, angle)

            # Zoom In/Out
            if zoom_func is not None:
                evframes, labels = zoom_func(evframes, labels, zoom_factor, center=zoom_center)

            # Add preprocessed frame and labels to the list
            augmented_evframes_list.append(evframes)
            augmented_labels_list.append(labels)

        # Return the results as a dictionary
        outputs = {
            'events': augmented_evframes_list,
            'labels': augmented_labels_list,
            'img_size': inputs['img_size'],
            'is_first_sample': inputs['is_first_sample'],     
            'is_padded_mask': inputs['is_padded_mask']   # Padding mask
        }

        return outputs