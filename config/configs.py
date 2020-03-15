import colorsys


class Config(object):
    # the default input of the image
    input_dim = 600
    # define the size of anchor_box
    anchor_box_scales = [128, 256, 512]
    # the ratio of height to width, e.g. 128-128, 128-256, 256-128
    anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
    # the stride of RPN, the magnification of the feature map with the raw input
    rpn_stride = 16
    # the batch size of a single classifier network prediction
    num_rois = 32
    # the path to the weight, default is None and can be changed
    model_path = None
    # the minimum magnification of the overlap when training RPN network, below which the area will not be encounter.
    rpn_min_overlap = 0.3
    # the maximum magnification of the overlap when training RPN network, above which the area will be encounter.
    rpn_max_overlap = 0.7
    # the minimum magnification of the overlap when training classifier network,
    # below which the area will not be encounter.
    classifier_min_overlap = 0.1
    # the maximum magnification of the overlap when training classifier network,
    # above which the area will be encounter.
    classifier_max_overlap = 0.5
    # TODO: Add meaning to classifier_regr_std
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
    # the name of all the classes, use for train and predict
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # the reserve entries in RPN result
    rpn_result_batch = 300
    # the threshold of IOU in RPN result when running NMS
    iou_threshold = 0.7
    # identifier threshold
    identifier_threshold = 0.3
    # identifier threshold for NMS
    identifier_threshold_nms = 0.4
