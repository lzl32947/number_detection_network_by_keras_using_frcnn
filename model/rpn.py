from keras import Model
from keras.layers import *

from model.resnet_50 import ResNet50


def rpn_net(inputs, num_anchors):
    feature_map = ResNet50(inputs)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        feature_map)

    # 生成9类的分类值
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 生成4x9的回归值
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, feature_map]


def rpn_model():
    inputs = Input(shape=(None, None, 3))
    output = rpn_net(inputs, 9)
    model = Model(inputs=inputs, outputs=output)
    return model
