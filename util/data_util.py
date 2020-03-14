import numpy as np
import pdb
import math
import copy
import time
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
from random import shuffle
import random
from PIL import Image
from keras.objectives import categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from config.configs import Config
from util.anchors import get_anchors
from util.encode_util import assign_boxes


def rand(a=0, b=1):
    """
    :param a:lower number
    :param b:upper number
    :return: random number between a and b
    """
    return np.random.rand() * (b - a) + a


def union(au, bu, area_intersection):
    """
    This function return the total area of the two rectangle
    """
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """
    This function return the area of inter section of two rectangle
    """
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """
    This function return the iou of two rectangle.
    """
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    # float, and avoid divided by zero
    return float(area_i) / float(area_u + 1e-6)


def calc_iou(R, all_boxes, width, height, num_classes):
    """
    :param R: the output of the RPN network[:,1:], with array of (rpn_result_batch,4)
    :param all_boxes: the boxes from the input, with shape (targets_number, 5), and in the last dimension the 5
                        stands for the (xmin, ymin, xmax, ymax, classification result)
    :param width: the width of input
    :param height: the height of input
    :param num_classes: the count of classes
    :return:
    """
    rpn_stride = Config.rpn_stride
    bboxes = all_boxes[:, :4]
    gta = np.zeros((len(bboxes), 4))
    # get the result of the box in feature map
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[0] * width / rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1] * height / rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2] * width / rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3] * height / rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []
    # go through the boxes from the RPN result and find the most match one with the given target area.
    # 遍历每一个由RPN网络产生的候选框并且找到与输入图像IOU最大的输入框
    for ix in range(R.shape[0]):
        x1 = R[ix, 0] * width / rpn_stride
        y1 = R[ix, 1] * height / rpn_stride
        x2 = R[ix, 2] * width / rpn_stride
        y2 = R[ix, 3] * height / rpn_stride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        # find the best place that have the max iou from the results of RPN network(rpn_result_batch,4)
        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        # not suitable
        if best_iou < 0.1:
            continue
        else:
            # find one
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)
            # iou between 0.1 and 0.5
            if 0.1 <= best_iou < 0.5:
                # the negative samples
                label = -1
            elif 0.5 <= best_iou:
                # the positive samples
                label = int(all_boxes[best_bbox, -1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0
                # standardize the input result
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # generate label
        class_label = num_classes * [0]
        class_label[label] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (num_classes - 1)
        labels = [0] * 4 * (num_classes - 1)
        if label != -1:
            # the positive samples
            label_pos = 4 * label
            sx, sy, sw, sh = Config.classifier_regr_std
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            # the negative samples
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        # No result
        return None, None, None, None
    # X with shape (?, 4)
    X = np.array(x_roi)
    # Y1 with shape (?,num_class)
    Y1 = np.array(y_class_num)
    # TODO: The meaning of Y2
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def get_new_img_size(width, height, img_min_side=Config.input_dim):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width), get_output_length(height)


def generate(line, n_classes, solid=True, solid_shape=[600, 600]):
    train_lines = line
    train_batches = len(line)
    num_classes = n_classes
    solid = solid
    solid_shape = solid_shape
    while True:
        shuffle(train_lines)
        lines = train_lines
        for annotation_line in lines:
            img, y = get_random_data(annotation_line)
            height, width, _ = np.shape(img)

            if len(y) == 0:
                continue
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / width
            boxes[:, 1] = boxes[:, 1] / height
            boxes[:, 2] = boxes[:, 2] / width
            boxes[:, 3] = boxes[:, 3] / height

            box_heights = boxes[:, 3] - boxes[:, 1]
            box_widths = boxes[:, 2] - boxes[:, 0]
            if (box_heights <= 0).any() or (box_widths <= 0).any():
                continue

            y[:, :4] = boxes[:, :4]

            anchors = get_anchors(get_img_output_length(width, height), width, height)

            # 计算真实框对应的先验框，与这个先验框应当有的预测结果
            assignment = assign_boxes(y, anchors)

            num_regions = 256

            classification = assignment[:, 4]
            regression = assignment[:, :]

            mask_pos = classification[:] > 0
            num_pos = len(classification[mask_pos])
            if num_pos > num_regions / 2:
                val_locs = random.sample(range(num_pos), int(num_pos - num_regions / 2))
                classification[mask_pos][val_locs] = -1
                regression[mask_pos][val_locs, -1] = -1

            mask_neg = classification[:] == 0
            num_neg = len(classification[mask_neg])
            if len(classification[mask_neg]) + num_pos > num_regions:
                val_locs = random.sample(range(num_neg), int(num_neg - num_pos))
                classification[mask_neg][val_locs] = -1

            classification = np.reshape(classification, [-1, 1])
            regression = np.reshape(regression, [-1, 5])

            tmp_inp = np.array(img)
            tmp_targets = [np.expand_dims(np.array(classification, dtype=np.float32), 0),
                           np.expand_dims(np.array(regression, dtype=np.float32), 0)]

            yield preprocess_input(np.expand_dims(tmp_inp, 0)), tmp_targets, np.expand_dims(y, 0)


def get_random_data(annotation_line, solid=True, solid_shape=[600, 600], random=True, jitter=.1, hue=.1, sat=1.1,
                    val=1.1, proc_img=True):
    '''r实时数据增强的随机预处理'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    if solid:
        w, h = solid_shape
    else:
        w, h = get_new_img_size(iw, ih)
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.9, 1.1)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1
    # correct boxes
    box_data = np.zeros((len(box), 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data[:len(box)] = box
    if len(box) == 0:
        return image_data, []
    if (box_data[:, :4] > 0).any():
        return image_data, box_data
    else:
        return image_data, []
