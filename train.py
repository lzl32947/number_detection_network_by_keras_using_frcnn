import os
from datetime import datetime

import keras.backend as K
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from config.Configs import Config, PMethod
from models import init_session, RPN_model, Classifier_model
import numpy as np
import tensorflow as tf
import keras

from util.input_util import get_anchors, rpn_data_generator, classifier_data_generator


def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels = y_true
        anchor_state = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # 找出存在目标的先验框
        indices_for_object = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
        normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

        normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
        normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0], keras.backend.floatx())
        normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object) / normalizer_pos
        cls_loss_for_back = ratio * keras.backend.sum(cls_loss_for_back) / normalizer_neg

        # 总的loss
        loss = cls_loss_for_object + cls_loss_for_back

        return loss

    return _cls_loss


def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1


def rpn_train():
    init_session()
    anchors = get_anchors(
        (np.ceil(Config.input_dim / Config.rpn_stride), np.ceil(Config.input_dim / Config.rpn_stride)),
        (Config.input_dim, Config.input_dim), Config.anchor_box_scales, Config.anchor_box_ratios, Config.rpn_stride)
    rpn_model = RPN_model([os.path.join(Config.weight_dir, "voc_weights.h5"), ], show_image=False)
    rpn_model.compile(loss={
        'regression': smooth_l1(),
        'classification': cls_loss()
    }, optimizer=keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])

    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(Config.checkpoint_dir, time)
    log_dir = os.path.join(Config.tensorboard_log_dir, time)
    os.mkdir(checkpoint_dir)
    os.mkdir(log_dir)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, 'rpn_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-rpn.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    rpn_model.fit_generator(
        rpn_data_generator(Config.train_annotation_path, anchors, batch_size=2, method=PMethod.Reshape),
        steps_per_epoch=1000,
        epochs=100,
        validation_data=rpn_data_generator(Config.valid_annotation_path, anchors, batch_size=2,
                                           method=PMethod.Reshape),
        validation_steps=100,
        initial_epoch=0,
        callbacks=[logging, checkpoint],
        verbose=1)


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])
        return loss

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def classifier_train():
    init_session()
    classifier_model = Classifier_model([os.path.join(Config.weight_dir, "voc_weights.h5"), ], show_image=False)
    classifier_model.compile(loss=[
        class_loss_cls,
        class_loss_regr(len(Config.class_names))
    ],
        metrics={"classification": 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(Config.checkpoint_dir, time)
    log_dir = os.path.join(Config.tensorboard_log_dir, time)
    os.mkdir(checkpoint_dir)
    os.mkdir(log_dir)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, 'classifier_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-classifier.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    classifier_model.fit_generator(
        classifier_data_generator(Config.train_annotation_path, method=PMethod.Reshape),
        steps_per_epoch=1000,
        epochs=100,
        validation_data=classifier_data_generator(Config.valid_annotation_path,
                                                  method=PMethod.Reshape),
        validation_steps=100,
        initial_epoch=0,
        callbacks=[logging, checkpoint],
        verbose=1)


if __name__ == '__main__':
    classifier_train()
