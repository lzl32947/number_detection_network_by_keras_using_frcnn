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

    detection_result_dir = "data/result"

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]


class model_config(Config):
    def __init__(self,
                 input_dim,
                 anchor_box_scales,
                 anchor_box_ratios,
                 rpn_stride,
                 feature_map_size,
                 num_rois,
                 rpn_min_overlap,
                 rpn_max_overlap,
                 classifier_overlap,
                 classifier_variance,
                 rpn_result_batch,
                 iou_threshold,
                 identifier_threshold,
                 identifier_threshold_nms,
                 classifier_train_batch,
                 rpn_variance,
                 feature_map_filters,
                 pooling_region,
                 freeze_layers,
                 ):
        self._input_dim = input_dim
        self._anchor_box_scales = anchor_box_scales
        self._anchor_box_ratios = anchor_box_ratios
        self._rpn_stride = rpn_stride
        self._num_rois = num_rois
        self._rpn_min_overlap = rpn_min_overlap
        self._rpn_max_overlap = rpn_max_overlap
        self._classifier_overlap = classifier_overlap
        self._classifier_variance = classifier_variance
        self._rpn_result_batch = rpn_result_batch
        self._iou_threshold = iou_threshold
        self._identifier_threshold = identifier_threshold
        self._identifier_threshold_nms = identifier_threshold_nms
        self._classifier_train_batch = classifier_train_batch
        self._rpn_variance = rpn_variance
        self._feature_map_filters = feature_map_filters
        self._pooling_region = pooling_region
        self._freeze_layers = freeze_layers
        self._feature_map_size = feature_map_size

    @property
    def freeze_layers(self):
        return self._freeze_layers

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def feature_map_size(self):
        return self._feature_map_size

    @property
    def anchor_box_scales(self):
        return self._anchor_box_scales

    @property
    def anchor_box_ratios(self):
        return self._anchor_box_ratios

    @property
    def rpn_stride(self):
        return self._rpn_stride

    @property
    def num_rois(self):
        return self._num_rois

    @property
    def rpn_min_overlap(self):
        return self._rpn_min_overlap

    @property
    def rpn_max_overlap(self):
        return self._rpn_max_overlap

    @property
    def classifier_overlap(self):
        return self._classifier_overlap

    @property
    def classifier_variance(self):
        return self._classifier_variance

    @property
    def rpn_result_batch(self):
        return self._rpn_result_batch

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @property
    def identifier_threshold(self):
        return self._identifier_threshold

    @property
    def identifier_threshold_nms(self):
        return self._identifier_threshold_nms

    @property
    def classifier_train_batch(self):
        return self._classifier_train_batch

    @property
    def rpn_variance(self):
        return self._rpn_variance

    @property
    def feature_map_filters(self):
        return self._feature_map_filters

    @property
    def pooling_region(self):
        return self._pooling_region


class ModelConfig(Enum):
    VGG16_standard = model_config(
        input_dim=600,
        anchor_box_scales=[64, 128, 256],
        anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
        rpn_stride=16,
        num_rois=32,
        rpn_min_overlap=0.3,
        rpn_max_overlap=0.7,
        classifier_overlap=0.5,
        classifier_variance=[8.0, 8.0, 4.0, 4.0],
        rpn_result_batch=300,
        iou_threshold=0.7,
        identifier_threshold=0.1,
        identifier_threshold_nms=0.4,
        classifier_train_batch=32,
        rpn_variance=[4, 4, 4, 4],
        feature_map_filters=512,
        pooling_region=7,
        freeze_layers=6,
        feature_map_size=37,
    )
    VGG16 = model_config(
        input_dim=600,
        anchor_box_scales=[64, 128, 256],
        anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
        rpn_stride=16,
        num_rois=32,
        rpn_min_overlap=0.3,
        rpn_max_overlap=0.7,
        classifier_overlap=0.5,
        classifier_variance=[8.0, 8.0, 4.0, 4.0],
        rpn_result_batch=300,
        iou_threshold=0.7,
        identifier_threshold=0.1,
        identifier_threshold_nms=0.4,
        classifier_train_batch=32,
        rpn_variance=[4, 4, 4, 4],
        feature_map_filters=512,
        pooling_region=14,
        freeze_layers=37,
        feature_map_size=37,
    )
    ResNet50 = model_config(
        input_dim=600,
        anchor_box_scales=[64, 128, 256],
        anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
        rpn_stride=16,
        num_rois=32,
        rpn_min_overlap=0.3,
        rpn_max_overlap=0.7,
        classifier_overlap=0.5,
        classifier_variance=[8.0, 8.0, 4.0, 4.0],
        rpn_result_batch=300,
        iou_threshold=0.7,
        identifier_threshold=0.1,
        identifier_threshold_nms=0.4,
        classifier_train_batch=32,
        rpn_variance=[4, 4, 4, 4],
        feature_map_filters=1024,
        pooling_region=14,
        freeze_layers=37,
        feature_map_size=38,
    )
    ResNet101 = model_config(
        input_dim=600,
        anchor_box_scales=[64, 128, 256],
        anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
        rpn_stride=16,
        num_rois=32,
        rpn_min_overlap=0.3,
        rpn_max_overlap=0.7,
        classifier_overlap=0.5,
        classifier_variance=[8.0, 8.0, 4.0, 4.0],
        rpn_result_batch=300,
        iou_threshold=0.7,
        identifier_threshold=0.1,
        identifier_threshold_nms=0.4,
        classifier_train_batch=32,
        rpn_variance=[4, 4, 4, 4],
        feature_map_filters=1024,
        pooling_region=14,
        freeze_layers=37,
        feature_map_size=38,
    )
    MobileNetV2 = model_config(
        input_dim=600,
        anchor_box_scales=[64, 128, 256],
        anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
        rpn_stride=16,
        num_rois=32,
        rpn_min_overlap=0.3,
        rpn_max_overlap=0.7,
        classifier_overlap=0.5,
        classifier_variance=[8.0, 8.0, 4.0, 4.0],
        rpn_result_batch=300,
        iou_threshold=0.7,
        identifier_threshold=0.1,
        identifier_threshold_nms=0.4,
        classifier_train_batch=32,
        rpn_variance=[4, 4, 4, 4],
        feature_map_filters=1280,
        pooling_region=7,
        freeze_layers=37,
        feature_map_size=37,
    )


class ImageProcessMethod(Enum):
    Zoom = 0
    Reshape = 1
