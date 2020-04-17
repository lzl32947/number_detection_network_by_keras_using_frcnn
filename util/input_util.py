import random
import tensorflow as tf
from config.Configs import Config, PMethod
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image

from util.image_util import zoom_image, resize_image


def calculate_iou(pos, anchors):
    """
    Return the IoU of the single box with anchor
    :param pos: numpy array, typically with shape (1,4)
    :param anchors: numpy array
    :return: numpy array, IoU result
    """
    pri = anchors
    _pos = np.expand_dims(pos, axis=0).repeat(len(pri), axis=0)
    inter_width = np.minimum(_pos[:, 2], pri[:, 2]) - np.maximum(_pos[:, 0], pri[:, 0])
    inter_height = np.minimum(_pos[:, 3], pri[:, 3]) - np.maximum(_pos[:, 1], pri[:, 1])
    inter_area = np.maximum(0, inter_height) * np.maximum(0, inter_width)
    full_area = (_pos[:, 3] - _pos[:, 1]) * (_pos[:, 2] - _pos[:, 0]) + (pri[:, 3] - pri[:, 1]) * (
            pri[:, 2] - pri[:, 0])
    iou = inter_area / (full_area - inter_area)
    return iou


def pos2label(loc, cls, anc):
    encoded_box = np.zeros((5,))
    box_center = 0.5 * (loc[:2] + loc[2:])
    box_wh = loc[2:] - loc[:2]

    assigned_priors_center = 0.5 * (anc[:2] +
                                    anc[2:4])
    assigned_priors_wh = (anc[2:4] -
                          anc[:2])

    encoded_box[:2] = box_center - assigned_priors_center
    encoded_box[:2] /= assigned_priors_wh
    encoded_box[:2] *= 4
    encoded_box[2:4] = np.log(box_wh / assigned_priors_wh)
    encoded_box[2:4] *= 4
    return encoded_box


def encode_box(box, anchor, variance):
    """
    Encode the single box to offset format.
    :param variance: tuple, variance used, with shape (4,)
    :param box: numpy array, the real box with format (x_min,y_min,x_max,y_max)
    :param anchor: numpy array
    :return: the encoded box
    """
    result = np.zeros(shape=(4,), dtype=np.float)
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]

    anchor_center = 0.5 * (anchor[:2] + anchor[2:4])
    anchor_wh = (anchor[2:4] - anchor[:2])

    result[:2] = box_center - anchor_center
    result[:2] /= anchor_wh

    result[2:4] = np.log(box_wh / anchor_wh)
    result[0] *= variance[0]
    result[1] *= variance[1]
    result[2] *= variance[2]
    result[3] *= variance[3]

    return result


def pos2area(input_pos, input_dim, rpn_stride):
    """
    Reformat the ROIs, for ROI was decoded to (x_min,y_min,x_max,y_max), transform it into (x_min, y_min, w, h),
    which should be used in ROI Pooling layers
    :param input_pos: numpy array, ROIs
    :param input_dim: int, the dimension of input image
    :param rpn_stride: int, the proportion of the input dimension and the feature map
    :return: numpy array, with format of (x_min, y_min, w, h)
    """
    input_pos[:, 0] = np.array(np.round(input_pos[:, 0] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 1] = np.array(np.round(input_pos[:, 1] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 2] = np.array(np.round(input_pos[:, 2] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 3] = np.array(np.round(input_pos[:, 3] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 2] -= input_pos[:, 0]
    input_pos[:, 3] -= input_pos[:, 1]
    return input_pos


def generate_anchors(sizes, ratios):
    """
    Function for generating anchors, to create the raw anchor from scale and aspect ratio.
    :param sizes: tuple, the scale of anchors
    :param ratios: tuple, the aspect ratio of anchors
    :return: numpy array, the raw anchor
    """
    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, rpn_strides):
    """
    Function for generating anchors.
    :param shape: tuple, the shape of the feature map
    :param anchors: tuple, the original anchor in the image
    :param rpn_strides: int, the proportion of the input dimension and the feature map
    :return: the anchor list
    """
    shift_x = (np.arange(0, shape[0], dtype=K.floatx()) + 0.5) * rpn_strides
    shift_y = (np.arange(0, shape[1], dtype=K.floatx()) + 0.5) * rpn_strides
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]
    k = np.shape(shifts)[0]
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),
                                                                                dtype=np.float)
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(feature_map_shape, input_shape, anchor_sizes, anchor_ratios, rpn_strides):
    """
    Generate the anchor for the RPN
    :param feature_map_shape: tuple, indicates the height and width of the feature map
    :param input_shape: tuple, the input dimension of the image
    :param anchor_sizes: tuple, the scale of the anchors
    :param anchor_ratios: tuple, the aspect ratio of the anchors
    :param rpn_strides: int, the proportion of the input dimension and the feature map
    :return: the anchor list
    """
    shape = feature_map_shape
    height, width = input_shape
    anchors = generate_anchors(anchor_sizes, anchor_ratios)
    network_anchors = shift(shape, anchors, rpn_strides)
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors


def process_pixel(img_array):
    """
    The same function of the keras.image.preprocess_input
    :param img_array: numpy array, the image content
    :return: the processed numpy array
    """
    img_array = img_array[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array


def process_input_image(image_path, method=PMethod.Zoom):
    """
    Process the input image.
    :param image_path: str, the path to original image
    :param method: PMethod class
    :return: numpy array of the processed image
    """
    image = Image.open(image_path)
    shape = np.array(image).shape
    if method == PMethod.Zoom:
        new_image = zoom_image(image)
    elif method == PMethod.Reshape:
        new_image = resize_image(image)
    else:
        raise RuntimeError("No Method Selected.")

    input_array = np.array(new_image, dtype=np.float)
    input_array = process_pixel(input_array)
    return input_array, shape


def image_show(image, li):
    """
    Show the rectangle of the object, use for test.
    :param image: PIL.Image object
    :param li: list or numpy array, must have shape (?,4) or (?,5)
    :return: None
    """
    plt.figure()
    plt.imshow(image.reshape(Config.input_dim, Config.input_dim, 3))
    for p in li:
        plt.gca().add_patch(
            plt.Rectangle((p[0], p[1]), p[2] - p[0],
                          p[3] - p[1], fill=False,
                          edgecolor='r', linewidth=1)
        )
    plt.show()
    plt.close()


def transform_box(box, original_shape, method):
    """
    Change the coordinate from the original annotation.
    :param box: numpy array, with shape (?,5)
    :param original_shape: tuple, indicates the (x_min,y_min,x_max,y_max) of the annotation
    :param method: PMethod class
    :return: the encoded boxes, with format (x_min,y_min,x_max,y_max)
    """
    h, w, c = original_shape
    if method == PMethod.Reshape:
        box = box.astype(np.float)
        box[:, 1:4:2] = box[:, 1:4:2] / h * Config.input_dim
        box[:, 0:4:2] = box[:, 0:4:2] / w * Config.input_dim
        box = box.astype(np.int)
    elif method == PMethod.Zoom:
        rh = Config.input_dim / h
        rw = Config.input_dim / w
        ranges = min(rw, rh)
        x_min_b = np.where(box[:, 0] >= 0.5 * Config.input_dim)
        x_min_l = np.where(box[:, 0] < 0.5 * Config.input_dim)
        x_max_b = np.where(box[:, 2] >= 0.5 * Config.input_dim)
        x_max_l = np.where(box[:, 2] < 0.5 * Config.input_dim)
        y_min_b = np.where(box[:, 1] >= 0.5 * Config.input_dim)
        y_min_l = np.where(box[:, 1] < 0.5 * Config.input_dim)
        y_max_b = np.where(box[:, 3] >= 0.5 * Config.input_dim)
        y_max_l = np.where(box[:, 3] < 0.5 * Config.input_dim)
        box[x_min_b, 0] = (Config.input_dim * 0.5 + (box[x_min_b, 0] - w * 0.5) * ranges).astype(np.int)
        box[x_min_l, 0] = (Config.input_dim * 0.5 - (w * 0.5 - box[x_min_l, 0]) * ranges).astype(np.int)
        box[x_max_b, 2] = (Config.input_dim * 0.5 + (box[x_max_b, 2] - w * 0.5) * ranges).astype(np.int)
        box[x_max_l, 2] = (Config.input_dim * 0.5 - (w * 0.5 - box[x_max_l, 2]) * ranges).astype(np.int)
        box[y_min_b, 1] = (Config.input_dim * 0.5 + (box[y_min_b, 1] - h * 0.5) * ranges).astype(np.int)
        box[y_min_l, 1] = (Config.input_dim * 0.5 - (h * 0.5 - box[y_min_l, 1]) * ranges).astype(np.int)
        box[y_max_b, 3] = (Config.input_dim * 0.5 + (box[y_max_b, 3] - h * 0.5) * ranges).astype(np.int)
        box[y_max_l, 3] = (Config.input_dim * 0.5 - (h * 0.5 - box[y_max_l, 3]) * ranges).astype(np.int)
    else:
        raise RuntimeError("No Method Selected.")
    return box


def generate_random_position(input_dim, rpn_stride):
    x = random.randint(0, np.ceil(input_dim / rpn_stride) - 1)
    y = random.randint(0, np.ceil(input_dim / rpn_stride) - 1)
    x_ = random.randint(x + 1, np.ceil(input_dim / rpn_stride))
    y_ = random.randint(y + 1, np.ceil(input_dim / rpn_stride))
    return [x, y, x_, y_]


def encode_label_for_rpn(pos, anchor):
    """
    This function generate the label Y for training of a single input image.
    :param pos: numpy array, the position of boxes
    :param anchor: numpy array
    :return: Y1: the confidence of the box to be ROI
    Y2: the encoded array of the boxes
    """
    loc = pos[:, :4].astype(np.float64) / Config.input_dim
    results = np.zeros(shape=(len(anchor), 6))
    results[:, 0] = -1
    negative_set = np.array([i for i in range(0, len(anchor))])
    for i in range(0, len(loc)):
        iou = calculate_iou(loc[i], anchor)

        positive_index = iou > Config.rpn_max_overlap
        k = np.argwhere(positive_index is True).tolist()
        if len(k) == 0:
            max_index = np.argmax(iou)
            k.append(max_index)

        for j in k:
            box = encode_box(loc[i], anchor[j], variance=Config.rpn_variance)
            if results[j, 5] < iou[j]:
                results[j, 0:4] = box
                results[j, 4] = 1
                results[j, 5] = iou[j]

        negative_index = np.where(iou < Config.rpn_min_overlap)
        negative_set = np.intersect1d(negative_set, negative_index)
    results[negative_set, 4] = 0
    return np.expand_dims(results[:, 4], -1), results[:, 0:5]


def encode_label_for_classifier(image_boxes):
    box_list = []
    gt_list = []
    label_list = []
    for i in image_boxes:
        x_min = int(round(i[0] / Config.rpn_stride))
        y_min = int(round(i[1] / Config.rpn_stride))
        x_max = int(round(i[2] / Config.rpn_stride))
        y_max = int(round(i[3] / Config.rpn_stride))
        box_list.append([x_min, y_min, x_max, y_max])
        gt_list.append([i[0] / Config.input_dim,
                        i[1] / Config.input_dim,
                        i[2] / Config.input_dim,
                        i[3] / Config.input_dim])
        label_list.append(i[4])
    if len(box_list) % Config.classifier_train_batch != 0:
        t = len(box_list) // Config.classifier_train_batch
        batch = (t + 1) * Config.classifier_train_batch
        for j in range(0, batch - len(box_list)):
            position = generate_random_position(Config.input_dim, Config.rpn_stride)
            box_list.append(position)
            label_list.append(0)

    X2 = np.array(box_list)
    X2[:, 2] = X2[:, 2] - X2[:, 0]
    X2[:, 3] = X2[:, 3] - X2[:, 1]
    box_list = np.array(box_list).astype(np.float)
    gt_list = np.array(gt_list).astype(np.float)
    box_list[:, 0] /= np.ceil(Config.input_dim / Config.rpn_stride)
    box_list[:, 1] /= np.ceil(Config.input_dim / Config.rpn_stride)
    box_list[:, 2] /= np.ceil(Config.input_dim / Config.rpn_stride)
    box_list[:, 3] /= np.ceil(Config.input_dim / Config.rpn_stride)
    Y1 = np.zeros(shape=(len(box_list), 1 + len(Config.class_names)), dtype=np.int)
    Y2 = np.zeros(shape=(len(box_list), 8 * len(Config.class_names)))

    results = np.zeros(shape=(len(box_list), 6))
    for i in range(0, len(box_list)):
        iou = calculate_iou(box_list[i], gt_list)

        positive_index = iou > Config.rpn_max_overlap
        k = np.argwhere(positive_index > 0).tolist()
        if len(k) == 0:
            max_index = np.argmax(iou)
            k.append(max_index)

        for j in k:
            box = encode_box(np.squeeze(box_list[i]), np.squeeze(gt_list[j]), Config.classifier_variance)
            if results[j, 5] < iou[j]:
                results[j, 0:4] = box
                results[j, 4] = 1
                results[j, 5] = iou[j]
                Y1[i, label_list[i]] = 1
    index = np.argmax(Y1, axis=1) == 0
    Y1[index, -1] = 1

    for i in range(0, len(box_list)):
        if results[i, 4] == 1:
            Y2[i, 4 * label_list[i]:4 * (label_list[i] + 1)] = [1, 1, 1, 1]
            Y2[i, 4 * len(Config.class_names) + 4 * label_list[i]:4 * len(Config.class_names) + 4 * (
                    label_list[i] + 1)] = results[i, 0:4]

    X2 = np.reshape(X2, (-1, Config.classifier_train_batch, 4))
    Y1 = np.reshape(Y1, (-1, Config.classifier_train_batch, len(Config.class_names) + 1))
    Y2 = np.reshape(Y2, (-1, Config.classifier_train_batch, 8 * len(Config.class_names)))
    return X2, Y1, Y2


def classifier_data_generator(annotation_path, method=PMethod.Reshape):
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation_lines = f.readlines()
    np.random.shuffle(annotation_lines)
    while True:
        for term in annotation_lines:
            line = term.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            image, o_s = process_input_image(img_path, method)
            u = transform_box(img_box, o_s, method)
            roi, class_conf, pos = encode_label_for_classifier(u)
            yield (
                {'image': np.expand_dims(np.array(image), axis=0), 'roi': roi},
                {'classification_1': class_conf, 'regression_1': pos})


def rpn_data_generator(annotation_path, anchors, batch_size=4, method=PMethod.Reshape):
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation_lines = f.readlines()
    np.random.shuffle(annotation_lines)
    X = []
    flag_list = []
    pos_list = []
    count = 0
    while True:
        for term in annotation_lines:
            line = term.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            x, o_s = process_input_image(img_path, method)
            u = transform_box(img_box, o_s, method)
            flag, pos = encode_label_for_rpn(u, anchors)
            X.append(x)
            flag_list.append(flag)
            pos_list.append(pos)
            count += 1
            if count == batch_size:
                count = 0
                yield (
                    {'input_1': np.array(X)}, {'classification': np.array(flag_list), 'regression': np.array(pos_list)})
                X = []
                flag_list = []
                pos_list = []
