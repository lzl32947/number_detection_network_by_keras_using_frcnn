from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from config.Configs import Config, PModel
from model.base_VGG16 import vgg16
from model.layers.ROIPoolingConv import RoiPoolingConv
from model.base_resnet50 import resnet50


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                               padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                               kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c',
                        trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                      trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x


def classifier_net(feature_map, input_shape, roi_region, num_rois, nb_classes=11):
    pooling_regions = 14

    # proposal 层相当于 [base_layers,input_rois]
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([feature_map, roi_region])
    # 以下是对于feature做操作
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    # 输出bbox_pred
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='classification_1')(out)
    # 输出class_prob
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='regression_1')(out)
    return [out_class, out_regr]


def classifier_model(model_class):
    if model_class == PModel.ResNet50:
        feature_map = Input(shape=(None, None, 1024), name="feature_map")
        input_shape = (Config.classifier_train_batch, 14, 14, 1024)
    elif model_class == PModel.VGG16:
        Config.rpn_stride = 32
        feature_map = Input(shape=(None, None, 512), name="feature_map")
        input_shape = (Config.classifier_train_batch, 14, 14, 512)
    else:
        raise RuntimeError("No model selected.")
    roi_region = Input(shape=(None, 4), name="roi")
    out = classifier_net(feature_map, input_shape, roi_region, Config.classifier_train_batch)
    model_classifier = Model(inputs=[feature_map, roi_region], outputs=out)
    return model_classifier


def classifier_model_for_train(model_class):
    image_input = Input(shape=(None, None, 3), name="image")
    roi_region = Input(shape=(None, 4), name="roi")
    if model_class == PModel.ResNet50:
        input_shape = (Config.classifier_train_batch, 14, 14, 1024)
        feature_map = resnet50(image_input)
    elif model_class == PModel.VGG16:
        input_shape = (Config.classifier_train_batch, 14, 14, 1024)
        feature_map = vgg16(image_input)
        Config.rpn_stride = 32
    else:
        raise RuntimeError("No model selected.")
    out = classifier_net(feature_map, input_shape, roi_region, Config.classifier_train_batch)
    model_classifier = Model(inputs=[image_input, roi_region], outputs=out)
    return model_classifier