from config.Configs import Config, ImageProcessMethod
import numpy as np
import tensorflow as tf
import keras.backend as K


def decode_classifier_result(model_name, cls, regr, roi, classifier_variance):
    """
    Decode the classifier model result to boxes.
    :param model_name:
    :param classifier_variance: tuple, the variance of classifier
    :param cls: numpy array, the result for classification from the classifier model
    :param regr: numpy array, the result for regression from the classifier model
    :param roi: numpy array, the ROIs from the RPN model
    :return: box with format (xmin,ymin,xmax,ymax),class,label
    """
    models = model_name.value
    label = np.argmax(cls)
    (x, y, w, h) = roi
    # transform x,y,w,h into float (0,1)
    x = x / models.feature_map_size
    y = y / models.feature_map_size
    w = w / models.feature_map_size
    h = h / models.feature_map_size

    cls_num = np.argmax(cls)
    (tx, ty, tw, th) = regr[4 * cls_num:4 * (cls_num + 1)]
    tx /= classifier_variance[0]
    ty /= classifier_variance[1]
    tw /= classifier_variance[2]
    th /= classifier_variance[3]

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

    return [x1, y1, x2, y2], np.max(cls), label


def decode_boxes_for_rpn(predict_loc, anchor, variance):
    """
    Reformat the prediction position from (cx,cy,w,h) to (x_min,y_min,x_max,y_max)
    :param variance: tuple with size 4, the variance for rpn
    :param predict_loc: tuple, indicates prediction position
    :param anchor: tuple, indicates anchor
    :return: tuple, the position of the decoded box
    """
    anchor_width = anchor[:, 2] - anchor[:, 0]
    anchor_height = anchor[:, 3] - anchor[:, 1]

    anchor_center_x = 0.5 * (anchor[:, 2] + anchor[:, 0])
    anchor_center_y = 0.5 * (anchor[:, 3] + anchor[:, 1])

    decode_box_center_x = predict_loc[:, 0] * anchor_width / variance[0]
    decode_box_center_x += anchor_center_x
    decode_box_center_y = predict_loc[:, 1] * anchor_height / variance[1]
    decode_box_center_y += anchor_center_y

    decode_box_width = np.exp(predict_loc[:, 2] / variance[2])
    decode_box_width *= anchor_width
    decode_box_height = np.exp(predict_loc[:, 3] / variance[3])
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


def nms_for_out(all_labels, all_confs, all_bboxes, num_classes, nms, rpn_result_batch):
    """
    NMS for the classifier model.
    :param rpn_result_batch: int, the batch size of nms
    :param all_labels: numpy array, the array of all result that decoded from classifier model
    :param all_confs: numpy array, the confidence of all result that decoded from classifier model
    :param all_bboxes: numpy array, the boxes of all result that decoded from classifier model
    :param num_classes: int, the length of all classes
    :param nms: float in range (0,1), the threshold for NMS
    :return: the list of box, confidence and label
    """
    results = []
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    nms_out = tf.image.non_max_suppression(boxes, scores,
                                           rpn_result_batch,
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


def rpn_output(predictions, anchors, top_k, confidence_threshold, rpn_result_batch, iou_threshold, rpn_variacne):
    """
    This function decode the output of RPN and return the ROIs of the original image.
    :param iou_threshold: float, the nms threshold for RPN
    :param rpn_result_batch: int, the batch size of rpn output
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
        decoded_positions = decode_boxes_for_rpn(predict_loc=box_loc[i],
                                                 anchor=anchors,
                                                 variance=rpn_variacne)
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
                                                                   rpn_result_batch,
                                                                   iou_threshold=iou_threshold),
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


def process_output_image(method, input_dim, original_shape, original_box, width_offset, height_offset):
    shape = original_shape
    box = original_box
    if method == ImageProcessMethod.Reshape:
        x_min = shape[1] * box[0]
        y_min = shape[0] * box[1]
        x_max = shape[1] * box[2]
        y_max = shape[0] * box[3]
    elif method == ImageProcessMethod.Zoom:
        x_min = input_dim * box[0]
        y_min = input_dim * box[1]
        x_max = input_dim * box[2]
        y_max = input_dim * box[3]
        if width_offset > 0:
            if x_min < input_dim * 0.5:
                x_min = input_dim * 0.5 - ((input_dim * 0.5 - x_min) * shape[0] / shape[1])
            else:
                x_min = input_dim * 0.5 + (x_min - input_dim * 0.5) * shape[0] / shape[1]
            if x_max < input_dim * 0.5:
                x_max = input_dim * 0.5 - ((input_dim * 0.5 - x_max) * shape[0] / shape[1])
            else:
                x_max = input_dim * 0.5 + (x_max - input_dim * 0.5) * shape[0] / shape[1]
        if height_offset > 0:
            if y_min < input_dim * 0.5:
                y_min = input_dim * 0.5 - ((input_dim * 0.5 - y_min) * shape[1] / shape[0])
            else:
                y_min = input_dim * 0.5 + (y_min - input_dim * 0.5) * shape[1] / shape[0]
            if y_max < input_dim * 0.5:
                y_max = input_dim * 0.5 - ((input_dim * 0.5 - y_max) * shape[1] / shape[0])
            else:
                y_max = input_dim * 0.5 + (y_max - input_dim * 0.5) * shape[1] / shape[0]
        x_min = x_min / input_dim * shape[1]
        x_max = x_max / input_dim * shape[1]
        y_min = y_min / input_dim * shape[0]
        y_max = y_max / input_dim * shape[0]
    else:
        raise RuntimeError("No Method Selected.")
    return x_min, y_min, x_max, y_max
