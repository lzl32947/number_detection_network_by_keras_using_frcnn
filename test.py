import colorsys

from keras_applications import imagenet_utils
from network.backbone.resnet50 import ResNet50
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.layers import *
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from network.classifier.resnet50_classifier import get_classifier
from network.rpn.common_rpn import get_rpn
from util.anchors import get_anchors
from util.decode_util import detection_out, nms_for_out

classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))


def get_img_output_length(width, height):
    def get_output_length(input_length):
        input_length += 6
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    net, result_layer = ResNet50()
    output_name = get_rpn(net, result_layer, 9)
    output_layer = [net[i] for i in output_name]
    model_rpn = Model(net['inputs'], output_layer)

    model_rpn.load_weights('weight/epoch027-loss0.328-rpn0.148-roi0.180.h5', by_name=True)

    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))
    classifier = get_classifier(feature_map_input, roi_input, 32, nb_classes=11, trainable=True)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_classifier.load_weights('weight/epoch027-loss0.328-rpn0.148-roi0.180.h5', by_name=True, skip_mismatch=True)

    image = Image.open(r"G:\data_stored\test_line\000651.jpg")
    image_shape = np.array(np.shape(image)[0:2])

    image = image.resize([600, 600])
    width = 600
    height = 600
    photo = np.array(image, dtype=np.float64)

    # 图片预处理，归一化
    photo = imagenet_utils.preprocess_input(np.expand_dims(photo, 0), data_format='channels_last')
    preds = model_rpn.predict(photo)
    # 将预测结果进行解码
    w_, h_ = get_img_output_length(width, height)
    anchors = get_anchors((w_, h_), width, height)

    rpn_results = detection_out(sess, preds, anchors, 1, confidence_threshold=0)
    R = rpn_results[0][:, 2:]

    R[:, 0] = np.array(np.round(R[:, 0] * width / 16), dtype=np.int32)
    R[:, 1] = np.array(np.round(R[:, 1] * height / 16), dtype=np.int32)
    R[:, 2] = np.array(np.round(R[:, 2] * width / 16), dtype=np.int32)
    R[:, 3] = np.array(np.round(R[:, 3] * height / 16), dtype=np.int32)

    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]
    base_layer = preds[2]

    bboxes = []
    probs = []
    labels = []
    for jk in range(R.shape[0] // 32 + 1):
        ROIs = np.expand_dims(R[32 * jk:32 * (jk + 1), :], axis=0)

        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // 32:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], 32, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([base_layer, ROIs])

        for ii in range(P_cls.shape[1]):
            if np.max(P_cls[0, ii, :]) < 0.3 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            label = np.argmax(P_cls[0, ii, :])

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])

            (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
            tx /= classifier_regr_std[0]
            ty /= classifier_regr_std[1]
            tw /= classifier_regr_std[2]
            th /= classifier_regr_std[3]

            cx = x + w / 2.
            cy = y + h / 2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy
            w1 = np.exp(tw) * w
            h1 = np.exp(th) * h

            x1 = cx1 - w1 / 2.
            y1 = cy1 - h1 / 2.

            x2 = cx1 + w1 / 2
            y2 = cy1 + h1 / 2

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            bboxes.append([x1, y1, x2, y2])
            probs.append(np.max(P_cls[0, ii, :]))
            labels.append(label)

    # 筛选出其中得分高于confidence的框
    labels = np.array(labels)
    probs = np.array(probs)
    boxes = np.array(bboxes, dtype=np.float32)
    boxes[:, 0] = boxes[:, 0] * 16 / width
    boxes[:, 1] = boxes[:, 1] * 16 / height
    boxes[:, 2] = boxes[:, 2] * 16 / width
    boxes[:, 3] = boxes[:, 3] * 16 / height
    results = np.array(
        nms_for_out(sess,np.array(labels), np.array(probs), np.array(boxes), 11 - 1, 0.4))

    if len(boxes) == 0:
        image.show()
    top_label_indices = results[:, 0]
    top_conf = results[:, 1]
    boxes = results[:, 2:]
    boxes[:, 0] = boxes[:, 0] * 600
    boxes[:, 1] = boxes[:, 1] * 600
    boxes[:, 2] = boxes[:, 2] * 600
    boxes[:, 3] = boxes[:, 3] * 600

    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

    thickness = (np.shape(image)[0] + np.shape(image)[1]) // width
    for i, c in enumerate(top_label_indices):
        predicted_class = class_names[int(c)]
        score = top_conf[i]

        left, top, right, bottom = boxes[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[int(c)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[int(c)])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    image.show()
