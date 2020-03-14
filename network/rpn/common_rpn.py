from keras import Model
from keras.layers import *
from keras.utils import plot_model

from network.backbone.resnet50 import ResNet50


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
