import os

import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model

from config.Configs import Config
from model.classifier import classifier_model
from model.rpn import rpn_model


def init_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)


def RPN_model(weight_file=None, show_image=False, show_summary=False):
    if weight_file is None:
        weight_file = []
    model = rpn_model()
    for i in weight_file:
        model.load_weights(i, skip_mismatch=True, by_name=True)
    if show_image:
        plot_model(model, to_file=os.path.join(Config.model_output_dir, "rpn.png"), show_shapes=True,
                   show_layer_names=True)
    if show_summary:
        model.summary()
    return model


def Classifier_model(weight_file=None, show_image=False, show_summary=False):
    if weight_file is None:
        weight_file = []
    model = classifier_model()
    for i in weight_file:
        model.load_weights(i, skip_mismatch=True, by_name=True)
    if show_image:
        plot_model(model, to_file=os.path.join(Config.model_output_dir, "classifier.png"), show_shapes=True,
                   show_layer_names=True)
    if show_summary:
        model.summary()
    return model
