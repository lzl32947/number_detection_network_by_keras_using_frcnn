import numpy as np
import tensorflow as tf

from config.configs import Config


def decode_boxes(mbox_loc, mbox_priorbox):
    # 获得先验框的宽与高
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

    # 获得先验框的中心点
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
    decode_bbox_center_y += prior_center_y

    # 真实框的宽与高的求取
    decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
    decode_bbox_height *= prior_height

    # 获取真实框的左上角与右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    # 真实框的左上角与右下角进行堆叠
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                  decode_bbox_ymin[:, None],
                                  decode_bbox_xmax[:, None],
                                  decode_bbox_ymax[:, None]), axis=-1)
    # 防止超出0与1
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox


def nms_for_out(sess, all_labels, all_confs, all_bboxes, num_classes, nms):
    results = []
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    nms_out = tf.image.non_max_suppression(boxes, scores,
                                           300,
                                           iou_threshold=nms)
    for c in range(num_classes):
        c_pred = []
        mask = all_labels == c
        if len(all_confs[mask]) > 0:
            # 取出得分高于confidence_threshold的框
            boxes_to_process = all_bboxes[mask]
            confs_to_process = all_confs[mask]
            # 进行iou的非极大抑制
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = sess.run(nms_out, feed_dict=feed_dict)
            # 取出在非极大抑制中效果较好的内容
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx][:, None]
            # 将label、置信度、框的位置进行堆叠。
            labels = c * np.ones((len(idx), 1))
            c_pred = np.concatenate((labels, confs, good_boxes), axis=1)
        results.extend(c_pred)
    return results


def rpn_output(sess, predictions, anchors, confidence_threshold=0):
    # 网络预测的结果
    # 置信度
    mbox_conf = predictions[0]
    mbox_loc = predictions[1]
    # 先验框
    results = []
    # 对每一个图片进行处理
    for i in range(len(mbox_loc)):
        k = []
        decode_bbox = decode_boxes(mbox_loc[i], anchors)
        c_confs = mbox_conf[i, :, 0]
        c_confs_m = c_confs > confidence_threshold
        if len(c_confs[c_confs_m]) > 0:
            # 取出得分高于confidence_threshold的框
            boxes_to_process = decode_bbox[c_confs_m]
            confs_to_process = c_confs[c_confs_m]
            boxes = tf.placeholder(dtype='float32', shape=(None, 4))
            scores = tf.placeholder(dtype='float32', shape=(None,))
            # 进行iou的非极大抑制
            feed_dict = {boxes: boxes_to_process,
                         scores: confs_to_process}
            idx = sess.run(tf.image.non_max_suppression(boxes, scores,
                                                           300,
                                                           iou_threshold=0.7), feed_dict=feed_dict)
            # 取出在非极大抑制中效果较好的内容
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx][:, None]
            # 将置信度、框的位置进行堆叠。
            c_pred = np.concatenate((confs, good_boxes),
                                    axis=1)
            # 添加进result里
            k.extend(c_pred)

        if len(k) > 0:
            # 按照置信度进行排序
            k = np.array(k)
            argsort = np.argsort(k[:, 1])[::-1]
            k = k[argsort]
            # 选出置信度最大的keep_top_k个
            k = k[:300]
        results.append(k)
    # 获得，在所有预测结果里面，置信度比较高的框
    # 还有，利用先验框和Retinanet的预测结果，处理获得了真实框（预测框）的位置
    return results
