import tensorflow as tf
import keras.backend as K
from PIL import Image
from keras import Model, Input

from config.configs import Config
from network.backbone.resnet50 import ResNet50
from network.classifier.resnet50_classifier import get_classifier, get_classifier_model
from network.rpn.common_rpn import get_rpn, get_rpn_model
from util.anchors import get_anchors
from util.decode_util import rpn_output, nms_for_out
from util.image_process_util import process_single_input

import numpy as np

from util.image_util import draw_result


def predict_images(image_list, configs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_rpn = get_rpn_model()
    model_rpn.load_weights(configs.model_path, by_name=True)
    model_classifier = get_classifier_model()
    model_classifier.load_weights(configs.model_path, by_name=True, skip_mismatch=True)

    anchors = get_anchors((38, 38), 600, 600)

    for file in image_list:
        image = Image.open(file)
        new_image, old_image = process_single_input(image, config=configs)
        # preds is the list of[classification, rpn_out_regress, feature_map]
        # the classification is weather the target is for GT, with shape(12996,1)
        # the rpn_out_regress is the target area, with shape(12996,4)
        # the feature_map is the compressed image, with shape(38,38,1024)
        preds = model_rpn.predict(new_image)

        # get the rpn result, with shape(5,300)
        # the [:,0] is for the possibility of the region to be selected.
        # the [:,1:] are for the regression of the target boundary.
        rpn_results = rpn_output(sess, preds, anchors)

        # change the result to fit the feature map size
        R = rpn_results[0][:, 1:]
        R[:, 0] = np.array(np.round(R[:, 0] * 600 / 16), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * 600 / 16), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * 600 / 16), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * 600 / 16), dtype=np.int32)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        # now R stands for the xmin, ymin, width, height of the selected region.

        base_layer = preds[2]
        bboxes = []
        probs = []
        labels = []
        # divide the 300 result to 32 to reduce the memory usage
        for jk in range(R.shape[0] // 32 + 1):
            ROIs = np.expand_dims(R[32 * jk:32 * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                # None left, break
                break

            if jk == R.shape[0] // 32:
                # This indicates that the batch is the last batch.
                # pad R with the first entry of this batch
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], 32, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            # P_cls has the shape (1,32,11)
            # P_regr has the shape (1,32,40)
            [P_cls, P_regr] = model_classifier.predict([base_layer, ROIs])

            for ii in range(P_cls.shape[1]):
                # if the area is the GT or has the greater chance to be the background then pass
                if np.max(P_cls[0, ii, :]) < 0.3 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # find one
                label = np.argmax(P_cls[0, ii, :])
                (x, y, w, h) = ROIs[0, ii, :]
                cls_num = np.argmax(P_cls[0, ii, :])
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= Config.classifier_regr_std[0]
                ty /= Config.classifier_regr_std[1]
                tw /= Config.classifier_regr_std[2]
                th /= Config.classifier_regr_std[3]

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

                # to generate the x_min,y_min,x_max,y_max in the feature map for running the NMS

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        # reset to decimal number for running the nms
        boxes[:, 0] = boxes[:, 0] * 16 / 600
        boxes[:, 1] = boxes[:, 1] * 16 / 600
        boxes[:, 2] = boxes[:, 2] * 16 / 600
        boxes[:, 3] = boxes[:, 3] * 16 / 600
        results = np.array(
            nms_for_out(sess, np.array(labels), np.array(probs), np.array(boxes), 11 - 1, 0.4))
        draw_result(old_image, results, boxes)
    sess.close()
