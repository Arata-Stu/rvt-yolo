def xywh2cxcywh(bboxes):
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5  # x + width / 2 = center_x
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5  # y + height / 2 = center_y
    return bboxes

