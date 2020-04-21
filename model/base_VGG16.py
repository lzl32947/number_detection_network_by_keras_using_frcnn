from keras.applications import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model


def vgg16(inputs):
    vgg16_builtin_model = VGG16(include_top=False,
                                weights='imagenet',
                                input_tensor=inputs)
    return vgg16_builtin_model.layers[-2].output


if __name__ == '__main__':
    plot_model(VGG16(include_top=False, weights='imagenet',
                     input_shape=(600, 600, 3)), to_file="./image/base_vgg16.png",
               show_layer_names=True, show_shapes=True)
