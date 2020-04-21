from enum import Enum


class Config(object):
    # 文件目录结构
    log_dir = "log"
    tensorboard_log_dir = "log/tensorboard"
    weight_dir = "log/weight"
    checkpoint_dir = "log/checkpoint"

    model_dir = "model"
    model_output_dir = "model/image"

    other_dir = "other"
    font_dir = "other/font"

    test_data_dir = "data/test"

    train_annotation_path = "data/train_annotation.txt"
    test_annotation_path = "data/test_annotation.txt"
    valid_annotation_path = "data/valid_annotation.txt"

    single_digits_dir = "data/single_digits"
    # the default input of the image
    input_dim = 600
    # define the size of anchor_box
    anchor_box_scales = [64, 128, 256]
    # the ratio of height to width, e.g. 128-128, 128-256, 256-128
    anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
    # the stride of RPN, the magnification of the feature map with the raw input
    rpn_stride = 16
    # the batch size of a single classifier network prediction
    num_rois = 32
    # the minimum magnification of the overlap when training RPN network, below which the area will not be encounter.
    rpn_min_overlap = 0.3
    # the maximum magnification of the overlap when training RPN network, above which the area will be encounter.
    rpn_max_overlap = 0.7
    # the maximum magnification of the overlap when training classifier network,
    # above which the area will be encounter.
    classifier_overlap = 0.5
    # the variance for classifier layers training
    classifier_variance = [8.0, 8.0, 4.0, 4.0]
    # the name of all the classes, use for train and predict
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # the reserve entries in RPN result
    rpn_result_batch = 300
    # the threshold of IOU in RPN result when running NMS
    iou_threshold = 0.7
    # identifier threshold
    identifier_threshold = 0.1
    # identifier threshold for NMS
    identifier_threshold_nms = 0.4
    # classifier train batch
    classifier_train_batch = 32
    # variance for rpn when encode box for training
    rpn_variance = [4, 4, 4, 4]


class PMethod(Enum):
    Zoom = 0
    Reshape = 1


class PModel(Enum):
    ResNet50 = "ResNet50"
    VGG16 = "VGG16"
    ResNet101 = "ResNet101"
    MobileNetV2 = "MobileNetV2"
    No = -1


class PClassifier(Enum):
    ResNetBase = "ResNetBase"
    DenseBase = "DenseBase"
    No = -1
