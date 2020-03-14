from keras import Model
from keras.layers import *
from keras.utils import plot_model

from layers.ROIPoolingConv import RoiPoolingConv
from network.backbone.resnet50 import ResNet50
from network.classifier.resnet50_classifier import classifier_layers
from network.rpn.common_rpn import get_rpn


def get_rpn_layers(inputs):
    net, result_layer = ResNet50(inputs)
    output_name = get_rpn(net, result_layer, 9)
    output_layer = [net[i] for i in output_name]
    return net, output_layer


def get_rpn_model(inputs):
    if inputs is None:
        inputs = Input(shape=(None, None, 3))
    net, result = get_rpn_layers(inputs)
    model_rpn = Model(net['inputs'], result)
    return model_rpn


def get_classifier_layers(feature_map_input, roi_input, num_rois, nb_classes, trainable):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    # proposal 层相当于 [base_layers,input_rois]
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([feature_map_input, roi_input])
    # 以下是对于feature做操作
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    # 输出bbox_pred
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # 输出class_prob
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def get_classifier_model(num_rois, nb_classes, trainable=False, feature_map_input=None, roi_input=None):
    if feature_map_input is None:
        feature_map_input = Input(shape=(None, None, 1024))
    if roi_input is None:
        roi_input = Input(shape=(None, 4))
    result = get_classifier_layers(feature_map_input, roi_input, num_rois, nb_classes, trainable)
    model = Model([feature_map_input, roi_input], result)
    return model


def get_all_model():
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    net, output_layer = get_rpn_layers(inputs)
    model_rpn = Model(inputs, output_layer[:2])

    classifier = get_classifier_layers(output_layer[2], roi_input, 32, nb_classes=11, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], output_layer[:2] + classifier)
    plot_model(model_all, show_shapes=True, show_layer_names=True)
    return model_rpn, model_classifier, model_all
