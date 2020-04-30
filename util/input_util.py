import os
import random
from config.Configs import Config, ImageProcessMethod, ModelConfig
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image

from util.image_generator import get_image_number_list, generate_single_image
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
    input_pos[:, 0] = np.array(np.floor(input_pos[:, 0] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 1] = np.array(np.floor(input_pos[:, 1] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 2] = np.array(np.ceil(input_pos[:, 2] * input_dim / rpn_stride), dtype=np.int32)
    input_pos[:, 3] = np.array(np.ceil(input_pos[:, 3] * input_dim / rpn_stride), dtype=np.int32)
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


def anchor_for_model(model_name):
    """
    Return the anchor of different models.
    :param model_name: ModelConfig class
    :return: numpy array, the anchor
    """
    models = model_name.value
    if model_name == ModelConfig.ResNet50:
        anchors = get_anchors(feature_map_shape=(models.feature_map_size, models.feature_map_size),
                              input_shape=(models.input_dim, models.input_dim),
                              anchor_sizes=models.anchor_box_scales,
                              anchor_ratios=models.anchor_box_ratios,
                              rpn_strides=models.rpn_stride)
    elif model_name == ModelConfig.VGG16_standard:
        anchors = get_anchors(feature_map_shape=(models.feature_map_size, models.feature_map_size),
                              input_shape=(models.input_dim, models.input_dim),
                              anchor_sizes=models.anchor_box_scales,
                              anchor_ratios=models.anchor_box_ratios,
                              rpn_strides=models.rpn_stride)
    elif model_name == ModelConfig.VGG16:
        anchors = get_anchors(feature_map_shape=(models.feature_map_size, models.feature_map_size),
                              input_shape=(models.input_dim, models.input_dim),
                              anchor_sizes=models.anchor_box_scales,
                              anchor_ratios=models.anchor_box_ratios,
                              rpn_strides=models.rpn_stride)
    else:
        raise RuntimeError("No model selected.")
    return anchors


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


def process_input_image(image, input_dim, method):
    """
    Process the input image.
    :param input_dim: int, the input dimension
    :param image: PIL.Image object
    :param method: PMethod class
    :return: numpy array of the processed image
    """
    shape = np.array(image).shape
    if method == ImageProcessMethod.Zoom:
        new_image = zoom_image(image, input_dim)
    elif method == ImageProcessMethod.Reshape:
        new_image = resize_image(image, input_dim)
    else:
        raise RuntimeError("No Method Selected.")

    input_array = np.array(new_image, dtype=np.float)
    input_array = process_pixel(input_array)
    return input_array, shape


def image_show(image, input_dim, li):
    """
    Show the rectangle of the object, use for test.
    :param input_dim: int, the input dimension
    :param image: PIL.Image object
    :param li: list or numpy array, must have shape (?,4) or (?,5)
    :return: None
    """
    plt.figure()
    plt.imshow(image.reshape(input_dim, input_dim, 3))
    for p in li:
        plt.gca().add_patch(
            plt.Rectangle((p[0], p[1]), p[2] - p[0],
                          p[3] - p[1], fill=False,
                          edgecolor='r', linewidth=1)
        )
    plt.show()
    plt.close()


def transform_box(box, input_dim, original_shape, method):
    """
    Change the coordinate from the original annotation.
    :param input_dim: int, the input dimension
    :param box: numpy array, with shape (?,5)
    :param original_shape: tuple, indicates the (x_min,y_min,x_max,y_max) of the annotation
    :param method: PMethod class
    :return: the encoded boxes, with format (x_min,y_min,x_max,y_max)
    """
    h, w, c = original_shape
    if method == ImageProcessMethod.Reshape:
        box = box.astype(np.float)
        box[:, 1:4:2] = box[:, 1:4:2] / h * input_dim
        box[:, 0:4:2] = box[:, 0:4:2] / w * input_dim
        box = box.astype(np.int)
    elif method == ImageProcessMethod.Zoom:
        rh = input_dim / h
        rw = input_dim / w
        ranges = min(rw, rh)
        x_min_b = np.where(box[:, 0] >= 0.5 * input_dim)
        x_min_l = np.where(box[:, 0] < 0.5 * input_dim)
        x_max_b = np.where(box[:, 2] >= 0.5 * input_dim)
        x_max_l = np.where(box[:, 2] < 0.5 * input_dim)
        y_min_b = np.where(box[:, 1] >= 0.5 * input_dim)
        y_min_l = np.where(box[:, 1] < 0.5 * input_dim)
        y_max_b = np.where(box[:, 3] >= 0.5 * input_dim)
        y_max_l = np.where(box[:, 3] < 0.5 * input_dim)
        box[x_min_b, 0] = (input_dim * 0.5 + (box[x_min_b, 0] - w * 0.5) * ranges).astype(np.int)
        box[x_min_l, 0] = (input_dim * 0.5 - (w * 0.5 - box[x_min_l, 0]) * ranges).astype(np.int)
        box[x_max_b, 2] = (input_dim * 0.5 + (box[x_max_b, 2] - w * 0.5) * ranges).astype(np.int)
        box[x_max_l, 2] = (input_dim * 0.5 - (w * 0.5 - box[x_max_l, 2]) * ranges).astype(np.int)
        box[y_min_b, 1] = (input_dim * 0.5 + (box[y_min_b, 1] - h * 0.5) * ranges).astype(np.int)
        box[y_min_l, 1] = (input_dim * 0.5 - (h * 0.5 - box[y_min_l, 1]) * ranges).astype(np.int)
        box[y_max_b, 3] = (input_dim * 0.5 + (box[y_max_b, 3] - h * 0.5) * ranges).astype(np.int)
        box[y_max_l, 3] = (input_dim * 0.5 - (h * 0.5 - box[y_max_l, 3]) * ranges).astype(np.int)
    else:
        raise RuntimeError("No Method Selected.")
    return box


def generate_random_position(feature_map_size):
    """
    Generate the random position for the pasted image.
    :param feature_map_size: int, the size of the feature map
    :return: the position
    """
    x = random.randint(0, feature_map_size - 1)
    y = random.randint(0, feature_map_size - 1)
    x_ = random.randint(x + 1, feature_map_size)
    y_ = random.randint(y + 1, feature_map_size)
    return [x, y, x_, y_]


def encode_label_for_rpn(pos, model_name, anchor):
    """
    This function generate the label Y for training of a single input image for the RPN model.
    :param model_name: ModelConfig object
    :param pos: numpy array, the position of boxes
    :param anchor: numpy array
    :return: Y1: the confidence of the box to be ROI
    Y2: the encoded array of the boxes
    """
    models = model_name.value
    loc = pos[:, :4].astype(np.float64) / models.input_dim
    # results:
    # 0,1,2,3 -> xmin, ymin, xmax, ymax
    # 4 -> label
    # 5 -> iou
    results = np.zeros(shape=(len(anchor), 6))
    # TODO: Check whether the change results[:,0] to results[:,4] is right
    results[:, 4] = -1
    negative_set = np.array([i for i in range(0, len(anchor))])
    for i in range(0, len(loc)):
        # the IoU of i-th location to the anchor
        iou = calculate_iou(loc[i], anchor)

        positive_index = iou > models.rpn_max_overlap
        k = np.argwhere(positive_index is True).tolist()
        if len(k) == 0:
            max_index = np.argmax(iou)
            k.append(max_index)

        for j in k:
            box = encode_box(loc[i], anchor[j], variance=models.rpn_variance)
            if results[j, 5] < iou[j]:
                results[j, 0:4] = box
                results[j, 4] = 1
                results[j, 5] = iou[j]

        negative_index = np.where(iou < models.rpn_min_overlap)
        negative_set = np.intersect1d(negative_set, negative_index)
    results[negative_set, 4] = 0
    return np.expand_dims(results[:, 4], -1), results[:, 0:5]


# TODO: Add function hint.
def encode_label_for_classifier(image_boxes, model_name):
    """
    This function generate the label Y for training of a single input image for the classifier model.
    :param model_name:
    :param image_boxes:
    :return: Y1: the confidence of the box to be ROI
    Y2: the encoded array of the boxes
    """
    models = model_name.value
    box_list = []
    gt_list = []
    label_list = []
    for i in image_boxes:
        # position for ROI:

        x_min = np.floor(i[0] / models.input_dim * models.feature_map_size)
        y_min = np.floor(i[1] / models.input_dim * models.feature_map_size)
        x_max = np.ceil(i[2] / models.input_dim * models.feature_map_size)
        y_max = np.ceil(i[3] / models.input_dim * models.feature_map_size)

        # box_list in format int  (0, feature map size)
        box_list.append([x_min, y_min, x_max, y_max])

        # gt_list in format float
        gt_list.append([i[0] / models.input_dim,
                        i[1] / models.input_dim,
                        i[2] / models.input_dim,
                        i[3] / models.input_dim])
        label_list.append(i[4])
    if len(box_list) > models.classifier_train_batch:
        box_list = box_list[:models.classifier_train_batch]
        gt_list = gt_list[:models.classifier_train_batch]
        label_list = label_list[:models.classifier_train_batch]
    else:
        batch = models.classifier_train_batch
        j = 0
        gt_box = np.array(box_list, dtype=np.int)
        while j < batch - len(gt_box):
            position = generate_random_position(models.feature_map_size)
            iou_list = calculate_iou(np.array(position) / models.feature_map_size, gt_box)
            if np.max(iou_list) > 0.3:
                continue
            else:
                box_list.append(position)
                label_list.append(0)
                j += 1
    # X2 in format int (0, feature map size)
    X2 = np.array(box_list)
    box_list = X2.copy()
    box_list = np.array(box_list).astype(np.float)
    # gt_list in format float (0, 1)
    gt_list = np.array(gt_list).astype(np.float)

    box_list[:, 0] = box_list[:, 0] / models.feature_map_size
    box_list[:, 1] = box_list[:, 1] / models.feature_map_size
    box_list[:, 2] = box_list[:, 2] / models.feature_map_size
    box_list[:, 3] = box_list[:, 3] / models.feature_map_size

    # box_list in format float (0, 1)

    X2[:, 2] = X2[:, 2] - X2[:, 0]
    X2[:, 3] = X2[:, 3] - X2[:, 1]

    Y1 = np.zeros(shape=(len(box_list), 1 + len(models.class_names)), dtype=np.int)
    Y2 = np.zeros(shape=(len(box_list), 8 * len(models.class_names)))

    results = np.zeros(shape=(len(box_list), 6))
    # results:
    # 0,1,2,3 -> xmin, ymin, xmax, ymax
    # 4 -> label
    # 5 -> iou
    for i in range(0, len(box_list)):
        iou = calculate_iou(box_list[i], gt_list)

        positive_index = iou > models.rpn_max_overlap
        k = np.argwhere(positive_index > 0).tolist()
        if len(k) == 0:
            max_index = np.argmax(iou)
            k.append(max_index)

        for j in k:
            box = encode_box(np.squeeze(gt_list[j]), np.squeeze(box_list[i]), models.classifier_variance)
            if results[j, 5] < iou[j]:
                results[j, 0:4] = box
                results[j, 4] = 1
                results[j, 5] = iou[j]
                Y1[i, label_list[i]] = 1
    index = np.max(Y1, axis=1) == 0
    Y1[index, -1] = 1

    for i in range(0, len(box_list)):
        if results[i, 4] == 1:
            Y2[i, 4 * label_list[i]:4 * (label_list[i] + 1)] = [1, 1, 1, 1]
            Y2[i, 4 * len(models.class_names) + 4 * label_list[i]:4 * len(models.class_names) + 4 * (
                    label_list[i] + 1)] = results[i, 0:4]

    X2 = np.reshape(X2, (-1, models.classifier_train_batch, 4))
    Y1 = np.reshape(Y1, (-1, models.classifier_train_batch, len(models.class_names) + 1))
    Y2 = np.reshape(Y2, (-1, models.classifier_train_batch, 8 * len(models.class_names)))
    return X2, Y1, Y2


def classifier_data_generator(annotation_path, model_name, method, use_generator=False):
    """
    The generator for training the classifier model.
    :param model_name: ModelConfig object
    :param annotation_path: str, the path to annotation file(if not using generator)
    :param method: PMethod class, the method to process the input image
    :param use_generator: bool, if use generator, the annotation file will not be used
    :return: single training data
    """
    models = model_name.value
    if use_generator:
        annotation_lines = []
        img_list = get_image_number_list()
    else:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_lines = f.readlines()
        np.random.shuffle(annotation_lines)
        img_list = []
    while True:
        if use_generator:
            img, box = generate_single_image(img_list)
            image, o_s = process_input_image(image=img,
                                             method=method,
                                             input_dim=models.input_dim)
            u = transform_box(box=box,
                              original_shape=o_s,
                              method=method,
                              input_dim=models.input_dim)
            roi, class_conf, pos = encode_label_for_classifier(image_boxes=u,
                                                               model_name=model_name)
            yield (
                {'input_1': np.expand_dims(np.array(image), axis=0), 'input_2': roi},
                {'dense_class_{}'.format(len(Config.class_names) + 1): class_conf,
                 'dense_regress_{}'.format(len(Config.class_names) + 1): pos})
        else:
            for term in annotation_lines:
                line = term.split()
                img_path = line[0]
                img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
                img = Image.open(img_path)
                image, o_s = process_input_image(image=img,
                                                 method=method,
                                                 input_dim=models.input_dim)
                u = transform_box(box=img_box,
                                  original_shape=o_s,
                                  method=method,
                                  input_dim=models.input_dim)
                roi, class_conf, pos = encode_label_for_classifier(image_boxes=u,
                                                                   model_name=model_name)
                yield (
                    {'input_1': np.expand_dims(np.array(image), axis=0), 'input_2': roi},
                    {'dense_class_{}'.format(len(Config.class_names) + 1): class_conf,
                     'dense_regress_{}'.format(len(Config.class_names) + 1): pos})


def rpn_data_generator(annotation_path, model_name, anchors, method, batch_size=4, use_generator=False):
    """
    The generator for training the RPN model.
    :param model_name: ModelConfig object
    :param batch_size: int, size of a batch
    :param anchors: numpy array
    :param annotation_path: str, the path to annotation file(if not using generator)
    :param method: PMethod class, the method to process the input image
    :param use_generator: bool, if use generator, the annotation file will not be used
    :return: single training data
    """
    models = model_name.value
    if use_generator:
        annotation_lines = []
        img_list = get_image_number_list()
    else:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_lines = f.readlines()
        np.random.shuffle(annotation_lines)
        img_list = []
    X = []
    flag_list = []
    pos_list = []
    count = 0
    while True:
        if use_generator:
            image, box = generate_single_image(img_list)
            x, o_s = process_input_image(image=image,
                                         method=method,
                                         input_dim=models.input_dim)
            u = transform_box(box=box,
                              original_shape=o_s,
                              method=method,
                              input_dim=models.input_dim)
            flag, pos = encode_label_for_rpn(pos=u,
                                             anchor=anchors,
                                             model_name=model_name)
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
        else:
            for term in annotation_lines:
                line = term.split()
                img_path = line[0]
                img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
                image = Image.open(img_path)
                x, o_s = process_input_image(image=image,
                                             method=method,
                                             input_dim=models.input_dim)
                u = transform_box(box=img_box,
                                  original_shape=o_s,
                                  method=method,
                                  input_dim=models.input_dim)
                flag, pos = encode_label_for_rpn(pos=u,
                                                 anchor=anchors,
                                                 model_name=model_name)
                X.append(x)
                flag_list.append(flag)
                pos_list.append(pos)
                count += 1
                if count == batch_size:
                    count = 0
                    yield (
                        {'input_1': np.array(X)},
                        {'classification': np.array(flag_list), 'regression': np.array(pos_list)})
                    X = []
                    flag_list = []
                    pos_list = []


def get_weight_file(name):
    """
    Return the weight file in the logs.
    :param name: str, the name of weight file
    :return: str, the path to selected weight file
    """
    file_list = []
    for root, dirs, files in os.walk(Config.checkpoint_dir):
        for d in dirs:
            dir_path = os.path.join(root, d)
            for _r, _d, f in os.walk(dir_path):
                for dx in _d:
                    _dir = os.path.join(dir_path, dx)
                    for _root, _x, weight_files in os.walk(_dir):
                        for weight_file in weight_files:
                            if weight_file == name:
                                file_list.append(os.path.join(_dir, weight_file))
                            elif name + ".h5" == weight_file:
                                file_list.append(os.path.join(_dir, weight_file))
    for root, dirs, files in os.walk(Config.weight_dir):
        for weight_file in files:
            if weight_file == name:
                file_list.append(os.path.join(root, weight_file))
            elif name + ".h5" == weight_file:
                file_list.append(os.path.join(root, weight_file))
    if len(file_list) == 0:
        raise RuntimeError("No weight suitable.")
    else:
        if len(file_list) == 1:
            print("Use weight in {}".format(file_list[0]))
            return file_list[0]
        else:
            recent_file = None
            recent_time = 0
            for f in file_list:
                time = os.path.getctime(f)
                if time > recent_time:
                    recent_file = f
                    recent_time = time
            if recent_file is not None:
                print("Use weight in {}".format(recent_file))
                return recent_file
            else:
                raise RuntimeError("No weight suitable.")
