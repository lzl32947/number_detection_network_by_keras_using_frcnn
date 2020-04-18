import os
import random
from PIL import Image
import numpy as np
from config.Configs import Config, PMethod
from models import RPN_model, Classifier_model, init_session
from util.image_generator import get_image_number_list, generate_single_image
from util.image_util import draw_image, draw_image_by_plt
from util.input_util import process_input_image, get_anchors, pos2area
from util.output_util import rpn_output, decode_classifier_result, nms_for_out
import matplotlib.pyplot as plt


def prediction_for_image():
    init_session()
    anchors = get_anchors(
        (np.ceil(Config.input_dim / Config.rpn_stride), np.ceil(Config.input_dim / Config.rpn_stride)),
        (Config.input_dim, Config.input_dim), Config.anchor_box_scales, Config.anchor_box_ratios, Config.rpn_stride)
    rpn_model = RPN_model(weight_file=[os.path.join(Config.checkpoint_dir,
                                                    "20200417_232828/rpn_ep005-loss0.221-val_loss0.151-rpn.h5"), ],
                          show_image=True)
    classifier_model = Classifier_model(for_train=False,
                                        weight_file=[os.path.join(Config.checkpoint_dir,
                                                                  "20200417_235634/classifier_ep025-loss0.097-val_loss0.022-classifier.h5"), ],
                                        show_image=True)
    # for root, dirs, files in os.walk(r"G:\data_stored\generated_train"):
    #     for image_file in files:
    #         path = os.path.join(root, image_file)
    #         image_list.append(path)

    # with open(Config.train_annotation_path, "r") as f:
    #     for line in f:
    #         image_list.append(line.split(" ")[0])
    # for f in image_list:
    img_list = get_image_number_list()
    while True:
        raw_image, _ = generate_single_image(img_list)
        image, shape = process_input_image(raw_image, PMethod.Reshape)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        res = rpn_output(predictions, anchors, Config.rpn_result_batch)[0]

        r = []
        for i in range(0, len(res)):
            r.append([res[i][1:5], res[i][0], 0])
        draw_image(raw_image.copy(), r, PMethod.Reshape, show_label=False)

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
            draw_image(raw_image, draw_list, PMethod.Reshape)
        else:
            draw_image(raw_image, [], PMethod.Reshape)


def prediction_for_analysis_generator(img_list, rpn_model, classifier_model, show_image=False):
    while True:
        raw_image, raw_list = generate_single_image(img_list)
        image, shape = process_input_image(raw_image, PMethod.Reshape)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        res = rpn_output(predictions, anchors, Config.rpn_result_batch)[0]

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
        result_list = []
        if len(box) > 0:
            box[:, 0] = box[:, 0] * Config.rpn_stride / Config.input_dim
            box[:, 1] = box[:, 1] * Config.rpn_stride / Config.input_dim
            box[:, 2] = box[:, 2] * Config.rpn_stride / Config.input_dim
            box[:, 3] = box[:, 3] * Config.rpn_stride / Config.input_dim
            results = nms_for_out(np.array(index), np.array(conf), np.array(box), len(Config.class_names),
                                  Config.identifier_threshold_nms)
            for item in range(0, len(results)):
                result_list.append([results[item][2:], results[item][1], results[item][0]])
            if show_image:
                draw_image_by_plt(raw_image, result_list, PMethod.Reshape)
        yield result_list, raw_list, shape


if __name__ == '__main__':
    init_session()
    anchors = get_anchors(
        (np.ceil(Config.input_dim / Config.rpn_stride), np.ceil(Config.input_dim / Config.rpn_stride)),
        (Config.input_dim, Config.input_dim), Config.anchor_box_scales, Config.anchor_box_ratios, Config.rpn_stride)
    model_rpn = RPN_model(weight_file=[os.path.join(Config.checkpoint_dir,
                                                    "20200417_232828/rpn_ep005-loss0.221-val_loss0.151-rpn.h5"), ],
                          show_image=True)
    model_classifier = Classifier_model(for_train=False,
                                        weight_file=[os.path.join(Config.checkpoint_dir,
                                                                  "20200417_235634/classifier_ep025-loss0.097-val_loss0.022-classifier.h5"), ],
                                        show_image=True)
    image_list = get_image_number_list()
    TP_list = [0] * len(Config.class_names)
    FP_list = [0] * len(Config.class_names)
    FN_list = [0] * len(Config.class_names)

    num_list = [[] for i in range(0, len(Config.class_names))]
    object_list = [0] * len(Config.class_names)
    ap_list = [0.0] * len(Config.class_names)


    def cross(pred_box, gt_box):
        cx = (pred_box[0] + pred_box[2]) / 2
        cy = (pred_box[1] + pred_box[3]) / 2
        if gt_box[0] < cx < gt_box[2] and gt_box[1] < cy < gt_box[3]:
            return 1
        else:
            return 0


    def voc_ap(rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return float(np.squeeze(ap))


    count = 0
    for prediction_result, gt, img_shape in prediction_for_analysis_generator(image_list, model_rpn, model_classifier,
                                                                              show_image=False):
        pred = np.zeros(shape=(len(prediction_result), 6), dtype=np.float)
        for i in range(0, len(prediction_result)):
            pred[i, 0] = int(prediction_result[i][0][0] * img_shape[1])
            pred[i, 1] = int(prediction_result[i][0][1] * img_shape[0])
            pred[i, 2] = int(prediction_result[i][0][2] * img_shape[1])
            pred[i, 3] = int(prediction_result[i][0][3] * img_shape[0])
            pred[i, 4] = int(prediction_result[i][2])
            pred[i, 5] = prediction_result[i][1]

        for num in range(0, len(Config.class_names)):
            TP = 0
            FP = 0
            gt_mask = np.argwhere(gt[:, 4] == num)
            gt_box = gt[gt_mask]
            object_list[num] += len(gt_box)
            FN_count = np.zeros(shape=(len(gt_mask),), dtype=np.int)

            pred_mask = np.argwhere(pred[:, 4] == num)
            for item in pred_mask:
                pred_box = pred[item]
                match = False
                for j in range(0, len(gt_box)):
                    if cross(np.squeeze(pred_box[0, :4]), np.squeeze(gt_box[j, :4])):
                        TP += 1
                        match = True
                        num_list[num].append((pred_box[0, 5], 1))
                        FN_count[j] = 1
                        break
                if not match:
                    FP += 1
                    num_list[num].append((pred_box[0, 5], 0))
            FN = len(np.argwhere(FN_count == 0))

            TP_list[num] += TP
            FP_list[num] += FP
            FN_list[num] += FN
            # print("current class:{}\tTP:{}\tFP:{}\tFN:{}".format(num, TP, FP, FN))
        count += 1
        if count % 50 == 0:
            print("finish {} count".format(count))
            plt.figure()
            plt.xlabel("Recall")
            plt.ylabel('Precision')
            plt.title("mAP for {}th image".format(count))
            for i in range(0, len(Config.class_names)):
                TP_ = TP_list[i]
                FP_ = FP_list[i]
                FN_ = FN_list[i]
                print("class:{}\tP:{:.2f}\tR:{:.2f}".format(i, TP_ / (TP_ + FP_), TP_ / (TP_ + FN_)), end="\t")
                k = np.array(num_list[i])
                k = k[np.argsort(-k[:, 0])]
                rec = []
                prec = []
                total = object_list[i]
                tp = 0
                fp = 0
                for j in range(0, len(k)):
                    if k[j, 1] > 0.5:
                        tp += 1
                    else:
                        fp += 1
                    prec.append(tp / (tp + fp))
                    rec.append(tp / total)
                r, g, b = random.random(), random.random(), random.random()
                plt.plot(rec, prec, linewidth=1, color=(r, g, b), marker='o', label="class {}".format(i))
                ap = voc_ap(rec, prec)
                ap_list[i] = ap
                print("AP:{:.4f}".format(ap))
            t = 0
            for i in ap_list:
                t += i
            print("total mAP for {}th:{:.4f}".format(count, t / len(Config.class_names)), end="\n\n")
            plt.legend()
            plt.show()
            plt.close()
        if count == 1000:
            break
