from keras import Model
from keras.layers import *
from keras.utils import plot_model

from network.backbone.resnet50 import ResNet50


def get_rpn(net, feature_map_name, num_anchors):
    base_layer = net[feature_map_name]
    # 首先进行3x3的卷积
    net['rpn_conv1'] = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal',
                              name='rpn_conv1')(
        base_layer)

    # 生成9类的分类值
    net['rpn_out_class'] = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                  name='rpn_out_class')(net['rpn_conv1'])
    # 生成4x9的回归值
    net['rpn_out_regress'] = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                                    name='rpn_out_regress')(net['rpn_conv1'])

    net['classification'] = Reshape((-1, 1), name="classification")(net['rpn_out_class'])
    net['rpn_out_regress'] = Reshape((-1, 4), name="regression")(net['rpn_out_regress'])

    # 返回的是分类层、回归层和原始层
    return ['classification', 'rpn_out_regress', feature_map_name]

