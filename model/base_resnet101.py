from keras import Input, Model
from keras.applications import ResNet101
from keras.utils import plot_model


def resnet101(inputs):
    resnet101_builtin_model = ResNet101(include_top=False,
                                        weights='imagenet',
                                        input_tensor=inputs)
    return resnet101_builtin_model.layers[-33].output



