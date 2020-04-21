from keras import Model
from keras.layers import *

from config.Configs import PModel, Config
from model.base_VGG16 import vgg16
from model.base_mobilenetv2 import mobilenetv2
from model.base_resnet101 import resnet101
from model.base_resnet50 import resnet50


def rpn_net(inputs, num_anchors, model_name):
    if model_name == PModel.ResNet50:
        feature_map = resnet50(inputs)
    elif model_name == PModel.ResNet101:
        feature_map = resnet101(inputs)
    elif model_name == PModel.VGG16:
        feature_map = vgg16(inputs)
        Config.rpn_stride = 16
    elif model_name == PModel.MobileNetV2:
        feature_map = mobilenetv2(inputs)
        Config.rpn_stride = 32
    else:
        raise RuntimeError("No model selected.")
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        feature_map)

    # 生成9类的分类值
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 生成4x9的回归值
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, feature_map]


def rpn_model(model_class):
    inputs = Input(shape=(None, None, 3))
    output = rpn_net(inputs, 9, model_class)
    model = Model(inputs=inputs, outputs=output)
    return model
