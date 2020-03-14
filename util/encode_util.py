import tensorflow as tf
import numpy as np

priors = None

def ciou(box):
    global priors
    # 计算出每个真实框与所有的先验框的iou
    # 判断真实框与先验框的重合情况
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:4], box[2:])

    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # 真实框的面积
    area_true = (box[2] - box[0]) * (box[3] - box[1])
    # 先验框的面积
    area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
    # 计算iou
    union = area_true + area_gt - inter

    iou = inter / union
    return iou


def encode_box(box, return_iou=True):
    iou = ciou(box)
    num_priors = 12996
    overlap_threshold = 0.7
    encoded_box = np.zeros((num_priors, 4 + return_iou))

    # 找到每一个真实框，重合程度较高的先验框
    assign_mask = iou > overlap_threshold
    if not assign_mask.any():
        assign_mask[iou.argmax()] = True
    if return_iou:
        encoded_box[:, -1][assign_mask] = iou[assign_mask]

    # 找到对应的先验框
    assigned_priors = priors[assign_mask]
    # 逆向编码，将真实框转化为Retinanet预测结果的格式
    # 先计算真实框的中心与长宽
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    # 再计算重合度较高的先验框的中心与长宽
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                    assigned_priors[:, 2:4])
    assigned_priors_wh = (assigned_priors[:, 2:4] -
                          assigned_priors[:, :2])

    # 逆向求取ssd应该有的预测结果
    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] *= 4

    encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
    encoded_box[:, 2:4][assign_mask] *= 4
    return encoded_box.ravel()


def ignore_box(box):
    iou = ciou(box)

    num_priors = 12996
    ignore_threshold =0.3
    overlap_threshold =0.7


    ignored_box = np.zeros((num_priors, 1))

    # 找到每一个真实框，重合程度较高的先验框
    assign_mask = (iou > ignore_threshold) & (iou < overlap_threshold)

    if not assign_mask.any():
        assign_mask[iou.argmax()] = True

    ignored_box[:, 0][assign_mask] = iou[assign_mask]
    return ignored_box.ravel()


def assign_boxes(boxes, anchors):
    global priors
    num_priors = len(anchors)
    priors = anchors
    assignment = np.zeros((num_priors, 4 + 1))

    assignment[:, 4] = 0.0
    if len(boxes) == 0:
        return assignment

    # 对每一个真实框都进行iou计算
    ingored_boxes = np.apply_along_axis(ignore_box, 1, boxes[:, :4])
    # 取重合程度最大的先验框，并且获取这个先验框的index
    ingored_boxes = ingored_boxes.reshape(-1, num_priors, 1)
    # (num_priors)
    ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
    # (num_priors)
    ignore_iou_mask = ignore_iou > 0

    assignment[:, 4][ignore_iou_mask] = -1

    # (n, num_priors, 5)
    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4])
    # 每一个真实框的编码后的值，和iou
    # (n, num_priors)
    encoded_boxes = encoded_boxes.reshape(-1, num_priors, 5)

    # 取重合程度最大的先验框，并且获取这个先验框的index
    # (num_priors)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    # (num_priors)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
    # (num_priors)
    best_iou_mask = best_iou > 0
    # 某个先验框它属于哪个真实框
    best_iou_idx = best_iou_idx[best_iou_mask]

    assign_num = len(best_iou_idx)
    # 保留重合程度最大的先验框的应该有的预测结果
    # 哪些先验框存在真实框
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]

    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    # 4代表为背景的概率，为0
    assignment[:, 4][best_iou_mask] = 1
    # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
    return assignment
