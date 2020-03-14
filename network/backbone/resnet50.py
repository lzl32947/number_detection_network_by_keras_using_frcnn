from keras import Model
from keras.layers import *
from keras.utils import plot_model
from config.configs import Config


def identity_block(net_dict, input_layer_name, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'activation' + str(stage) + block + '_branch'

    input_layer = net_dict[input_layer_name]

    net_dict[conv_name_base + '2a'] = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_layer)
    net_dict[bn_name_base + '2a'] = BatchNormalization(name=bn_name_base + '2a')(net_dict[conv_name_base + '2a'])
    net_dict[act_name_base + '2a'] = Activation('relu', name=act_name_base + '2a')(net_dict[bn_name_base + '2a'])

    net_dict[conv_name_base + '2b'] = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
        net_dict[act_name_base + '2a'])
    net_dict[bn_name_base + '2b'] = BatchNormalization(name=bn_name_base + '2b')(net_dict[conv_name_base + '2b'])
    net_dict[act_name_base + '2b'] = Activation('relu', name=act_name_base + '2b')(net_dict[bn_name_base + '2b'])

    net_dict[conv_name_base + '2c'] = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(
        net_dict[act_name_base + '2b'])
    net_dict[bn_name_base + '2c'] = BatchNormalization(name=bn_name_base + '2c')(net_dict[conv_name_base + '2c'])

    res = add([net_dict[bn_name_base + '2c'], input_layer])
    res_name_base = 'res' + str(stage) + block + '_branch'
    net_dict[res_name_base] = Activation('relu', name=res_name_base)(res)
    return net_dict, res_name_base


def conv_block(net_dict, input_layer_name, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'activation' + str(stage) + block + '_branch'

    input_layer = net_dict[input_layer_name]

    net_dict[conv_name_base + '2a'] = Conv2D(filters1, (1, 1), strides=strides,
                                             name=conv_name_base + '2a')(input_layer)
    net_dict[bn_name_base + '2a'] = BatchNormalization(name=bn_name_base + '2a')(net_dict[conv_name_base + '2a'])
    net_dict[act_name_base + '2a'] = Activation('relu', name=act_name_base + '2a')(net_dict[bn_name_base + '2a'])

    net_dict[conv_name_base + '2b'] = Conv2D(filters2, kernel_size, padding='same',
                                             name=conv_name_base + '2b')(net_dict[act_name_base + '2a'])
    net_dict[bn_name_base + '2b'] = BatchNormalization(name=bn_name_base + '2b')(net_dict[conv_name_base + '2b'])
    net_dict[act_name_base + '2n'] = Activation('relu', name=act_name_base + '2n')(net_dict[bn_name_base + '2b'])

    net_dict[conv_name_base + '2c'] = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(
        net_dict[act_name_base + '2n'])
    net_dict[bn_name_base + '2c'] = BatchNormalization(name=bn_name_base + '2c')(net_dict[conv_name_base + '2c'])

    net_dict[conv_name_base + '1'] = Conv2D(filters3, (1, 1), strides=strides,
                                            name=conv_name_base + '1')(input_layer)
    net_dict[bn_name_base + '1'] = BatchNormalization(name=bn_name_base + '1')(net_dict[conv_name_base + '1'])

    res = add([net_dict[bn_name_base + '2c'], net_dict[bn_name_base + '1']])
    res_name_base = 'res' + str(stage) + block + '_branch'
    net_dict[res_name_base] = Activation('relu', name=res_name_base)(res)
    return net_dict, res_name_base


def ResNet50(inputs):
    img_input = inputs
    net = {}
    net['inputs'] = img_input
    net['zero_padding_0'] = ZeroPadding2D((3, 3))(net['inputs'])
    net['conv1'] = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(net['zero_padding_0'])
    net['bn_conv1'] = BatchNormalization(name='bn_conv1')(net['conv1'])
    net['activation_0'] = Activation('relu')(net['bn_conv1'])

    net['max_pooling_0'] = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(net['activation_0'])

    net, layer_name = conv_block(net, 'max_pooling_0', 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    net, layer_name = identity_block(net, layer_name, 3, [64, 64, 256], stage=2, block='b')
    net, layer_name = identity_block(net, layer_name, 3, [64, 64, 256], stage=2, block='c')

    net, layer_name = conv_block(net, layer_name, 3, [128, 128, 512], stage=3, block='a')
    net, layer_name = identity_block(net, layer_name, 3, [128, 128, 512], stage=3, block='b')
    net, layer_name = identity_block(net, layer_name, 3, [128, 128, 512], stage=3, block='c')
    net, layer_name = identity_block(net, layer_name, 3, [128, 128, 512], stage=3, block='d')

    net, layer_name = conv_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='a')
    net, layer_name = identity_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='b')
    net, layer_name = identity_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='c')
    net, layer_name = identity_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='d')
    net, layer_name = identity_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='e')
    net, layer_name = identity_block(net, layer_name, 3, [256, 256, 1024], stage=4, block='f')

    return net, layer_name
