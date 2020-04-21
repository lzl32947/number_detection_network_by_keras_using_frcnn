from keras import Input, Model
from keras.applications import MobileNetV2
from keras.utils import plot_model


def mobilenetv2(inputs):
    mobilenetv2_builtin_model = MobileNetV2(include_top=False,
                                            weights='imagenet',
                                            input_tensor=inputs)
    return mobilenetv2_builtin_model.outputs[0]
