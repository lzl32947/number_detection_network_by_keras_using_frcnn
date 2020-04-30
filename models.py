from config.Configs import ModelConfig
from model.vgg16_standard import get_vgg16_standard_rpn, get_vgg16_standard_classifier, vgg16_base
from model.vgg16 import get_vgg16_rpn, get_vgg16_classifier
from model.vgg16 import vgg16_base as vg16_base
from model.resnet50 import resnet50_base, get_resnet50_rpn, get_resnet50_classifier
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
import tensorflow as tf
import os


def init_session(run_on_laptop=False, use_bfc=False):
    if run_on_laptop:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    if use_bfc:
        config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


def get_classifier_model(model_name):
    """
    Return the classifier model
    :param model_name: ModelConfig object
    :return: classifier model
    """
    models = model_name.value
    input_dim = models.input_dim
    num_roi = models.classifier_train_batch
    inputs = Input(shape=(input_dim, input_dim, 3))
    roi_input = Input(shape=(num_roi, 4))

    if model_name == ModelConfig.ResNet50:
        base_f = resnet50_base
        classifier_f = get_resnet50_classifier
    elif model_name == ModelConfig.VGG16_standard:
        base_f = vgg16_base
        classifier_f = get_vgg16_standard_classifier
    elif model_name == ModelConfig.VGG16:
        base_f = vg16_base
        classifier_f = get_vgg16_classifier
    else:
        raise RuntimeError("No model selected.")
    base_layers = base_f(inputs)

    classifier = classifier_f(base_layers, roi_input, num_roi)
    model_classifier = Model([inputs, roi_input], classifier)

    return model_classifier


def get_rpn_model(model_name):
    """
    Return the rpn model
    :param model_name: ModelConfig object
    :return: rpn model
    """
    models = model_name.value
    input_dim = models.input_dim
    inputs = Input(shape=(input_dim, input_dim, 3))
    num_anchors = len(models.anchor_box_scales) * len(models.anchor_box_ratios)

    if model_name == ModelConfig.ResNet50:
        base_f = resnet50_base
        rpn_f = get_resnet50_rpn
    elif model_name == ModelConfig.VGG16_standard:
        base_f = vgg16_base
        rpn_f = get_vgg16_standard_rpn
    elif model_name == ModelConfig.VGG16:
        base_f = vg16_base
        rpn_f = get_vgg16_rpn
    else:
        raise RuntimeError("No model selected.")
    base_layers = base_f(inputs)

    rpn = rpn_f(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    return model_rpn


def get_predict_model(model_name):
    """
    Return the models for prediction
    :param model_name: ModelConfig object
    :return: rpn model, classifier model
    """
    models = model_name.value
    input_dim = models.input_dim
    num_roi = models.classifier_train_batch
    feature_map_filters = models.feature_map_filters
    num_anchors = len(models.anchor_box_scales) * len(models.anchor_box_ratios)

    inputs = Input(shape=(input_dim, input_dim, 3))
    roi_input = Input(shape=(num_roi, 4))
    feature_map_input = Input(shape=(None, None, feature_map_filters))

    if model_name == ModelConfig.VGG16_standard:
        base_f = vgg16_base
        rpn_f = get_vgg16_standard_rpn
        classifier_f = get_vgg16_standard_classifier
    elif model_name == ModelConfig.ResNet50:
        base_f = resnet50_base
        rpn_f = get_resnet50_rpn
        classifier_f = get_resnet50_classifier
    elif model_name == ModelConfig.VGG16:
        base_f = vg16_base
        rpn_f = get_vgg16_rpn
        classifier_f = get_vgg16_classifier
    else:
        raise RuntimeError("No model selected.")

    base_layers = base_f(inputs)

    rpn = rpn_f(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = classifier_f(feature_map_input, roi_input, num_roi)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only
