from config.Configs import Config
import numpy as np
import tensorflow as tf
import keras.backend as K


# TODO: Rewrite the function
def decode_classifier_result(cls, regr, roi):
    label = np.argmax(cls)
    (x, y, w, h) = roi
    cls_num = np.argmax(cls)
    (tx, ty, tw, th) = regr[4 * cls_num:4 * (cls_num + 1)]
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
    return [x1, y1, x2, y2], np.max(cls), label


def decode_boxes(predict_loc, anchor):
    """
    Reformat the prediction position from (cx,cy,w,h) to (x_min,y_min,x_max,y_max)
    :param predict_loc: tuple, indicates prediction position
    :param anchor: tuple, indicates anchor
    :return: tuple, the position of the decoded box
    """
    anchor_width = anchor[:, 2] - anchor[:, 0]
    anchor_height = anchor[:, 3] - anchor[:, 1]

    anchor_center_x = 0.5 * (anchor[:, 2] + anchor[:, 0])
    anchor_center_y = 0.5 * (anchor[:, 3] + anchor[:, 1])

    decode_box_center_x = predict_loc[:, 0] * anchor_width / 4
    decode_box_center_x += anchor_center_x
    decode_box_center_y = predict_loc[:, 1] * anchor_height / 4
    decode_box_center_y += anchor_center_y

    decode_box_width = np.exp(predict_loc[:, 2] / 4)
    decode_box_width *= anchor_width
    decode_box_height = np.exp(predict_loc[:, 3] / 4)
    decode_box_height *= anchor_height

    decode_box_xmin = decode_box_center_x - 0.5 * decode_box_width
    decode_box_ymin = decode_box_center_y - 0.5 * decode_box_height
    decode_box_xmax = decode_box_center_x + 0.5 * decode_box_width
    decode_box_ymax = decode_box_center_y + 0.5 * decode_box_height

    decode_box = np.concatenate((decode_box_xmin[:, None],
                                 decode_box_ymin[:, None],
                                 decode_box_xmax[:, None],
                                 decode_box_ymax[:, None]), axis=-1)
    decode_box = np.minimum(np.maximum(decode_box, 0.0), 1.0)
    return decode_box


def nms_for_out(all_labels, all_confs, all_bboxes, num_classes, nms):
    results = []
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    nms_out = tf.image.non_max_suppression(boxes, scores,
                                           Config.rpn_result_batch,
                                           iou_threshold=nms)
    for c in range(num_classes):
        c_pred = []
        mask = np.where(all_labels == c)
        if len(all_confs[mask]) > 0:
            boxes_to_process = all_bboxes[mask]
            confs_to_process = all_confs[mask]
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = K.get_session().run(nms_out, feed_dict=feed_dict)
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx][:, None]
            labels = c * np.ones((len(idx), 1))
            c_pred = np.concatenate((labels, confs, good_boxes), axis=1)
        results.extend(c_pred)
    return results


def rpn_output(predictions, anchors, top_k, confidence_threshold=0.5):
    """
    This function decode the output of RPN and return the ROIs of the original image.
    :param top_k: int, the maximum of the reserved region
    :param predictions: the list of 3, indicates [conf_array,loc_array,feature_map], \
    conf_array with shape (batch size, anchor quantity, 1) \
    loc_array with shape (batch size, anchor quantity, 4)
    :param anchors: list or numpy array, indicates the anchor of the image
    :param confidence_threshold: float, the threshold above which the result will be saved
    :return: list, decoded boxes
    """
    box_conf = predictions[0]
    box_loc = predictions[1]
    batch_size = box_loc.shape[0]
    result = []
    for i in range(0, batch_size):
        decoded_positions = decode_boxes(box_loc[i], anchors)
        confidence = box_conf[i, :, 0]
        confidence_mask = np.where(confidence > confidence_threshold)[0]

        if len(confidence_mask) == 0:
            result.append(np.zeros(shape=(1, 5), dtype=np.float))
        else:
            boxes_to_process = decoded_positions[confidence_mask]
            confs_to_process = confidence[confidence_mask]
            boxes = tf.placeholder(dtype='float32', shape=(None, 4))
            scores = tf.placeholder(dtype='float32', shape=(None,))
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = K.get_session().run(tf.image.non_max_suppression(boxes, scores,
                                                                   Config.rpn_result_batch,
                                                                   iou_threshold=Config.iou_threshold),
                                      feed_dict=feed_dict)
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx]
            confs = np.reshape(confs, (len(confs), 1))
            raw_result = np.concatenate((confs, good_boxes), axis=1)
            sort_index = np.argsort(raw_result[:, 0])[::-1]
            single_result = raw_result[sort_index]
            single_result = single_result[:top_k]
            result.append(single_result)
    return result
