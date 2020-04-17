import os
import random

import numpy as np
from config.Configs import Config, PMethod
from models import RPN_model, Classifier_model, init_session
from util.image_util import draw_image
from util.input_util import process_input_image, get_anchors, pos2area
from util.output_util import rpn_output, decode_classifier_result, nms_for_out

if __name__ == '__main__':
    init_session()
    anchors = get_anchors(
        (np.ceil(Config.input_dim / Config.rpn_stride), np.ceil(Config.input_dim / Config.rpn_stride)),
        (Config.input_dim, Config.input_dim), Config.anchor_box_scales, Config.anchor_box_ratios, Config.rpn_stride)
    rpn_model = RPN_model(weight_file=[os.path.join(Config.checkpoint_dir,
                                                    "20200415_185557/rpn_ep005-loss1.270-val_loss1.319-anchor.h5"), ],
                          show_image=True)
    classifier_model = Classifier_model(for_train=False,
                                        weight_file=[os.path.join(Config.checkpoint_dir,
                                                                  "20200417_154117/classifier_ep009-loss0.657-val_loss0.778-classifier.h5"), ],
                                        show_image=True)
    image_list = []
    # for root, dirs, files in os.walk(r"G:\data_stored\generated_train"):
    #     for image_file in files:
    #         path = os.path.join(root, image_file)
    #         image_list.append(path)

    with open(Config.valid_annotation_path, "r") as f:
        for line in f:
            image_list.append(line.split(" ")[0])
    random.shuffle(image_list)
    for f in image_list:
        image, shape = process_input_image(f, PMethod.Reshape)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        res = rpn_output(predictions, anchors, Config.rpn_result_batch)[0]

        k = np.where(res[:, 0] > 0.5)
        r = []
        for item in k[0]:
            r.append([res[item][1:5], res[item][0], 0])
        draw_image(f, r, PMethod.Reshape, show_label=False)

        selected_area = pos2area(res[:, 1:], Config.input_dim, Config.rpn_stride)
        feature_map = predictions[2]

        box_list = []
        conf_list = []
        index_list = []
        for batch in range(selected_area.shape[0] // 32 + 1):
            ROIs = np.expand_dims(selected_area[32 * batch:32 * (batch + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break

            if batch == selected_area.shape[0] // 32:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], 32, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = classifier_model.predict([feature_map, ROIs])
            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < Config.identifier_threshold or np.argmax(P_cls[0, ii, :]) == (
                        P_cls.shape[2] - 1):
                    continue
                else:
                    box, conf, index = decode_classifier_result(P_cls[0, ii, :], P_regr[0, ii, :], ROIs[0, ii, :])
                    box_list.append(box)
                    conf_list.append(conf)
                    index_list.append(index)
        index = np.array(index_list)
        conf = np.array(conf_list)
        box = np.array(box_list, dtype=np.float32)
        draw_list = []
        if len(box) > 0:
            box[:, 0] = box[:, 0] * Config.rpn_stride / Config.input_dim
            box[:, 1] = box[:, 1] * Config.rpn_stride / Config.input_dim
            box[:, 2] = box[:, 2] * Config.rpn_stride / Config.input_dim
            box[:, 3] = box[:, 3] * Config.rpn_stride / Config.input_dim
            results = nms_for_out(np.array(index), np.array(conf), np.array(box), len(Config.class_names),
                                  Config.identifier_threshold_nms)
            for item in range(0, len(results)):
                draw_list.append([results[item][2:], results[item][1], results[item][0]])
            draw_image(f, draw_list, PMethod.Reshape)
        else:
            draw_image(f, [], PMethod.Reshape)
