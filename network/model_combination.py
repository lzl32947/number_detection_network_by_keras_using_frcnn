from keras import Model
from keras.layers import *
from keras.utils import plot_model

from layers.ROIPoolingConv import RoiPoolingConv
from network.backbone.resnet50 import ResNet50
from network.classifier.resnet50_classifier import classifier_layers
from network.rpn.common_rpn import get_rpn


def get_rpn(base_layers, num_anchors):
    # 首先进行3x3的卷积
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    # 生成9类的分类值
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 生成4x9的回归值
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)

    # 返回的是分类层、回归层和原始层
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    # proposal 层相当于 [base_layers,input_rois]
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
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


def get_model(num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    num_anchors = 9
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, 32, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all


def get_predict_model(num_classes):
    # inputs 是 输入的图像
    inputs = Input(shape=(None, None, 3))
    # roi_input 应该是输入的ROI
    roi_input = Input(shape=(None, 4))
    # feature map input 应该是经过resnet后输出的图像
    feature_map_input = Input(shape=(None, None, 1024))

    # base_layers 是输出的feature_map的层
    base_layers = ResNet50(inputs)
    # num_anchors 是 9
    num_anchors = 9
    # rpn 是 [分类层、回归层、原始的feature map层]
    rpn = get_rpn(base_layers, num_anchors)
    # rpn 模型以图像层作输入、rpn层作输出
    model_rpn = Model(inputs, rpn)

    # 分类层
    classifier = get_classifier(feature_map_input, roi_input, 32, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only


if __name__ == '__main__':
    m_r, m_c = get_predict_model(11)
    plot_model(m_c, to_file="classifier/classifier_model.png", show_layer_names=True, show_shapes=True)
