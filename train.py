from __future__ import division

from keras.losses import categorical_crossentropy
from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time
import tensorflow as tf
import keras.backend as K

from config.configs import Config
from network.model_combination import get_model
from util.data_util import get_anchors, get_random_data, get_data
from util.data_util import calc_iou, generate
from util.decode_util import rpn_output


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


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def get_img_output_length(width, height):
    def get_output_length(input_length):
        input_length += 6
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)

    EPOCH = 100
    EPOCH_LENGTH = 2000
    annotation_path = '2007_train.txt'
    NUM_CLASSES = len(Config.class_names) + 1
    model_rpn, model_classifier, model_all = get_model(len(Config.class_names) + 1)
    base_net_weights = "weight/voc_weights.h5"

    model_all.summary()
    model_rpn.load_weights(base_net_weights, by_name=True)
    model_classifier.load_weights(base_net_weights, by_name=True)

    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    rpn_train = generate(lines, NUM_CLASSES, data_function=get_data)
    log_dir = "logs"
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)

    model_rpn.compile(loss={
        'regression': smooth_l1(),
        'classification': cls_loss()
    }, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls,
        class_loss_regr(NUM_CLASSES - 1)
    ],
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    # 初始化参数
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')

    for i in range(EPOCH):

        if i == 20:
            model_rpn.compile(loss={
                'regression': smooth_l1(),
                'classification': cls_loss()
            }, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls,
                class_loss_regr(NUM_CLASSES - 1)
            ],
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")

        progbar = generic_utils.Progbar(EPOCH_LENGTH)
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, boxes = next(rpn_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)
            height, width, _ = np.shape(X[0])
            anchors = get_anchors(get_img_output_length(width, height), width, height)

            # 将预测结果进行解码
            results = rpn_output(sess, P_rpn, anchors, confidence_threshold=0)

            R = results[0][:, 1:]

            X2, Y1, Y2, IouS = calc_iou(R, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if len(neg_samples) == 0:
                continue
            num_rois = Config.num_rois
            if len(pos_samples) < num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, num_rois // 2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples),
                                                        replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples),
                                                        replace=True).tolist()

            sel_samples = selected_pos_samples + selected_neg_samples
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            train_step += 1
            iter_num += 1
            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if True:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                          ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                           'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                          [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                           loss_class_cls, loss_class_regr, class_acc, curr_loss], i)

                if True:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss, curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curr_loss,
                                                                                                      loss_rpn_cls + loss_rpn_regr,
                                                                                                      loss_class_cls + loss_class_regr) + ".h5")

                break
