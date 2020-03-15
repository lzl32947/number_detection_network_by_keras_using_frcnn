import numpy as np
import tensorflow as tf

from config.configs import Config


def decode_boxes(mbox_loc, mbox_priorbox):
    # get the width and height of prior box
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

    # get the center point of prior box
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # get the offset of the prior box and the real box
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
    decode_bbox_center_y += prior_center_y

    # get the width and height of the real box
    decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
    decode_bbox_height *= prior_height

    # get the top-left and bottom-right of the real box
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    # concatenate the result
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                  decode_bbox_ymin[:, None],
                                  decode_bbox_xmax[:, None],
                                  decode_bbox_ymax[:, None]), axis=-1)
    # clip the result by 0 and 1
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox


def nms_for_out(sess, all_labels, all_confs, all_bboxes, num_classes, nms):
    results = []
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    nms_out = tf.image.non_max_suppression(boxes, scores,
                                           Config.rpn_result_batch,
                                           iou_threshold=nms)
    # num_classes are the real class (exclude the GT)
    for c in range(num_classes):
        c_pred = []
        # select the all area that stands for c
        mask = all_labels == c
        # the result does have the label c
        if len(all_confs[mask]) > 0:
            # get the box that are higher than identifier_threshold_nms
            boxes_to_process = all_bboxes[mask]
            confs_to_process = all_confs[mask]
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = sess.run(nms_out, feed_dict=feed_dict)
            # get the better result in nms result
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx][:, None]
            # concatenate the label, confidence and location of the boxes
            labels = c * np.ones((len(idx), 1))
            c_pred = np.concatenate((labels, confs, good_boxes), axis=1)
        results.extend(c_pred)
    return results


def rpn_output(sess, predictions, anchors, confidence_threshold=0):
    # predictions[0] stands for the possibility of not GT
    mbox_conf = predictions[0]
    # predictions[1] stands for the boundary of the selected area
    mbox_loc = predictions[1]
    # the results list is needed if the input batch has more than one item
    results = []
    # process for each area
    for i in range(len(mbox_loc)):
        # i in range (0,12996)
        k = []
        # get the result of the x_min, y_min, x_max, y_max
        decode_bbox = decode_boxes(mbox_loc[i], anchors)
        c_confs = mbox_conf[i, :, 0]
        # c_confs_m stands for the list of [True, False]
        c_confs_m = c_confs > confidence_threshold
        # assert there are some selected area instead of 0, e.g. not all black
        if len(c_confs[c_confs_m]) > 0:
            # get the boxes that have higher confidence than threshold
            boxes_to_process = decode_bbox[c_confs_m]
            # and the confs
            confs_to_process = c_confs[c_confs_m]
            # prepare the NMS
            boxes = tf.placeholder(dtype='float32', shape=(None, 4))
            scores = tf.placeholder(dtype='float32', shape=(None,))
            # run the NMS for IOU to filtrate the repetitive boxes
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = sess.run(tf.image.non_max_suppression(boxes, scores,
                                                        Config.rpn_result_batch,
                                                        iou_threshold=Config.iou_threshold), feed_dict=feed_dict)
            # get the best result in the result
            good_boxes = boxes_to_process[idx]
            # and the confs
            confs = confs_to_process[idx][:, None]
            # concatenate the both
            c_pred = np.concatenate((confs, good_boxes),
                                    axis=1)
            # add to k
            k.extend(c_pred)

            if len(k) > 0:
                # 按照置信度进行排序
                k = np.array(k)
                argsort = np.argsort(k[:, 1])[::-1]
                k = k[argsort]
                # 选出置信度最大的keep_top_k个
                k = k[:Config.rpn_result_batch]
        results.append(k)

    return results
