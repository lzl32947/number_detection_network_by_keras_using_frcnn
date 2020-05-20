import os
import random
from datetime import datetime

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from config.Configs import ImageProcessMethod, Config, ModelConfig
from models import init_session, get_predict_model
from util.image_generator import get_image_number_list, generate_single_image
from util.image_util import draw_image_by_pillow, draw_image_by_plt
from util.input_util import anchor_for_model, process_input_image, pos2area, get_weight_file
from util.output_util import rpn_output, decode_classifier_result, nms_for_out


def prediction_for_image(rpn_weight_list, classifier_weight_list, model_name, method, use_generator=True,
                         use_annotation=False,
                         annotation_lines=None, run_on_laptop=False, use_bfc=True):
    init_session(run_on_laptop, use_bfc)
    models = model_name.value
    anchors = anchor_for_model(model_name)
    rpn_model, classifier_model = get_predict_model(model_name=model_name)

    for i in classifier_weight_list:
        classifier_model.load_weights(i, skip_mismatch=True, by_name=True)
    for i in rpn_weight_list:
        rpn_model.load_weights(i, skip_mismatch=True, by_name=True)

    img_list = get_image_number_list()
    while True:
        if use_generator:
            raw_image, _ = generate_single_image(img_list)
        elif use_annotation and annotation_lines is not None:
            line = annotation_lines.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            raw_image = Image.open(img_path)
        else:
            raise RuntimeError("No predict method selected.")
        image, shape = process_input_image(image=raw_image,
                                           method=method,
                                           input_dim=models.input_dim)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        res = rpn_output(predictions=predictions,
                         anchors=anchors,
                         top_k=models.rpn_result_batch,
                         confidence_threshold=0.5,
                         rpn_result_batch=models.rpn_result_batch,
                         iou_threshold=models.iou_threshold,
                         rpn_variacne=models.rpn_variance)[0]

        r = []
        for i in range(0, len(res)):
            r.append([res[i][1:5], res[i][0], 0])
        draw_image_by_pillow(image=raw_image.copy(),
                             input_dim=models.input_dim,
                             result_list=r,
                             method=ImageProcessMethod.Reshape,
                             show_label=False)

        selected_area = pos2area(res[:, 1:], models.input_dim, models.rpn_stride)
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
                if np.max(P_cls[0, ii, :]) < models.identifier_threshold or np.argmax(P_cls[0, ii, :]) == (
                        P_cls.shape[2] - 1):
                    continue
                else:
                    box, conf, index = decode_classifier_result(model_name=model_name,
                                                                cls=P_cls[0, ii, :],
                                                                regr=P_regr[0, ii, :],
                                                                roi=ROIs[0, ii, :],
                                                                classifier_variance=models.classifier_variance)
                    box_list.append(box)
                    conf_list.append(conf)
                    index_list.append(index)
        index = np.array(index_list)
        conf = np.array(conf_list)
        box = np.array(box_list, dtype=np.float32)
        draw_list = []
        if len(box) > 0:
            results = nms_for_out(all_labels=np.array(index),
                                  all_confs=np.array(conf),
                                  all_bboxes=np.array(box),
                                  num_classes=len(models.class_names),
                                  nms=models.identifier_threshold_nms,
                                  rpn_result_batch=models.rpn_result_batch)
            for item in range(0, len(results)):
                draw_list.append([results[item][2:], results[item][1], results[item][0]])
            draw_image_by_pillow(image=raw_image, input_dim=models.input_dim, result_list=draw_list,
                                 method=method)
        else:
            draw_image_by_pillow(image=raw_image, input_dim=models.input_dim, result_list=[], method=method)


def prediction_for_recording(record_name, rpn_weight_list, classifier_model_weight, model_name,
                             use_generator=True, generator_count=0, use_annotation=True, annotation_lines=None,
                             save_image=False, method=ImageProcessMethod.Reshape, run_on_laptop=False, use_bfc=True):
    models = model_name.value
    init_session(run_on_laptop, use_bfc)
    anchors = anchor_for_model(model_name)
    anchors = anchor_for_model(model_name)
    rpn_model, classifier_model = get_predict_model(model_name=model_name)

    for i in classifier_model_weight:
        classifier_model.load_weights(i, skip_mismatch=True, by_name=True)

    for i in rpn_weight_list:
        rpn_model.load_weights(i, skip_mismatch=True, by_name=True)
    image_list = get_image_number_list()

    if not os.path.exists(Config.detection_result_dir):
        os.mkdir(Config.detection_result_dir)
    record_dir = os.path.join(Config.detection_result_dir, record_name)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    time_dir = os.path.join(record_dir, time)
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    gt_dir = os.path.join(time_dir, "ground-truth")
    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)
    img_dir = os.path.join(time_dir, "images-optional")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    result_dir = os.path.join(time_dir, "detection-results")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    def predict_on_single_batch(image):
        image, shape = process_input_image(image=image,
                                           method=method,
                                           input_dim=models.input_dim)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        res = rpn_output(predictions=predictions,
                         anchors=anchors,
                         top_k=models.rpn_result_batch,
                         rpn_result_batch=models.rpn_result_batch,
                         iou_threshold=models.iou_threshold,
                         confidence_threshold=0.5,
                         rpn_variacne=models.rpn_variance)[0]

        selected_area = pos2area(res[:, 1:], models.input_dim, models.rpn_stride)
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
                if np.max(P_cls[0, ii, :]) < models.identifier_threshold or np.argmax(P_cls[0, ii, :]) == (
                        P_cls.shape[2] - 1):
                    continue
                else:
                    box, conf, index = decode_classifier_result(model_name=model_name,
                                                                cls=P_cls[0, ii, :],
                                                                regr=P_regr[0, ii, :],
                                                                roi=ROIs[0, ii, :],
                                                                classifier_variance=models.classifier_variance)
                    box_list.append(box)
                    conf_list.append(conf)
                    index_list.append(index)
        index = np.array(index_list)
        conf = np.array(conf_list)
        box = np.array(box_list, dtype=np.float32)
        result_list = []
        if len(box) > 0:

            results = nms_for_out(
                all_labels=np.array(index),
                all_confs=np.array(conf),
                all_bboxes=np.array(box),
                num_classes=len(Config.class_names),
                nms=models.identifier_threshold_nms,
                rpn_result_batch=models.rpn_result_batch)
            for item in range(0, len(results)):
                result_list.append([results[item][2:], results[item][1], results[item][0]])
        return result_list

    def decode_result_list(image, results):
        shape = np.array(image).shape
        width_zoom_ratio = models.input_dim / shape[1]
        height_zoom_ratio = models.input_dim / shape[0]
        zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
        new_width = int(zoom_ratio * shape[1])
        new_height = int(zoom_ratio * shape[0])

        width_offset = (models.input_dim - new_width) // 2
        height_offset = (models.input_dim - new_height) // 2
        width_offset /= models.input_dim
        height_offset /= models.input_dim
        for box, conf, index in results:
            if method == ImageProcessMethod.Reshape:
                x_min = shape[1] * box[0]
                y_min = shape[0] * box[1]
                x_max = shape[1] * box[2]
                y_max = shape[0] * box[3]
            elif method == ImageProcessMethod.Zoom:
                x_min = models.input_dim * box[0]
                y_min = models.input_dim * box[1]
                x_max = models.input_dim * box[2]
                y_max = models.input_dim * box[3]
                if width_offset > 0:
                    if x_min < models.input_dim * 0.5:
                        x_min = models.input_dim * 0.5 - (
                                (models.input_dim * 0.5 - x_min) * shape[0] / shape[1])
                    else:
                        x_min = models.input_dim * 0.5 + (x_min - models.input_dim * 0.5) * shape[0] / \
                                shape[1]
                    if x_max < models.input_dim * 0.5:
                        x_max = models.input_dim * 0.5 - (
                                (models.input_dim * 0.5 - x_max) * shape[0] / shape[1])
                    else:
                        x_max = models.input_dim * 0.5 + (x_max - models.input_dim * 0.5) * shape[0] / \
                                shape[1]
                if height_offset > 0:
                    if y_min < models.input_dim * 0.5:
                        y_min = models.input_dim * 0.5 - (
                                (models.input_dim * 0.5 - y_min) * shape[1] / shape[0])
                    else:
                        y_min = models.input_dim * 0.5 + (y_min - models.input_dim * 0.5) * shape[1] / \
                                shape[0]
                    if y_max < models.input_dim * 0.5:
                        y_max = models.input_dim * 0.5 - (
                                (models.input_dim * 0.5 - y_max) * shape[1] / shape[0])
                    else:
                        y_max = models.input_dim * 0.5 + (y_max - models.input_dim * 0.5) * shape[1] / \
                                shape[0]
                x_min = x_min / models.input_dim * shape[1]
                x_max = x_max / models.input_dim * shape[1]
                y_min = y_min / models.input_dim * shape[0]
                y_max = y_max / models.input_dim * shape[0]
            else:
                raise RuntimeError("No Method Selected.")
            yield int(index), conf, int(x_min), int(y_min), int(x_max), int(y_max)

    count = 1
    if use_annotation:
        for term in annotation_lines:
            line = term.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            img = Image.open(img_path)
            result_list = predict_on_single_batch(img)
            batch_name = "a{:0>4d}".format(count)
            with open(os.path.join(gt_dir, batch_name + ".txt"), "w") as f_gt:
                for line in img_box:
                    f_gt.write("{} {} {} {} {}\n".format(line[4], line[0], line[1], line[2], line[3]))
            with open(os.path.join(result_dir, batch_name + ".txt"), "w") as f_res:
                if len(result_list) > 0:
                    for i, c, xi, yi, xa, ya in decode_result_list(img, result_list):
                        f_res.write("{} {} {} {} {} {}\n".format(i, c, xi, yi, xa, ya))
            if save_image:
                img.save(os.path.join(img_dir, batch_name + ".png"))
            count += 1
            if count % 10 == 0:
                print("finish {} image for annotation".format(count))
    if use_generator:
        for count in range(1, generator_count + 1):
            raw_image, raw_list = generate_single_image(image_list)
            result_list = predict_on_single_batch(raw_image)
            batch_name = "g{:0>4d}".format(count)
            with open(os.path.join(gt_dir, batch_name + ".txt"), "w") as f_gt:
                for line in raw_list:
                    f_gt.write("{} {} {} {} {}\n".format(line[4], line[0], line[1], line[2], line[3]))
            with open(os.path.join(result_dir, batch_name + ".txt"), "w") as f_res:
                if len(result_list) > 0:
                    for i, c, xi, yi, xa, ya in decode_result_list(raw_image, result_list):
                        f_res.write("{} {} {} {} {} {}\n".format(i, c, xi, yi, xa, ya))
            if save_image:
                raw_image.save(os.path.join(img_dir, batch_name + ".png"))
            if count % 10 == 0:
                print("finish {} image for generator".format(count))


def prediction_for_analysis(rpn_weight_list, classifier_model_weight, model_name, method, run_on_laptop=False,
                            use_bfc=True):
    init_session(run_on_laptop, use_bfc)
    models = model_name.value
    anchors = anchor_for_model(model_name)
    rpn_model, classifier_model = get_predict_model(model_name=model_name)

    for i in classifier_model_weight:
        classifier_model.load_weights(i, skip_mismatch=True, by_name=True)

    for i in rpn_weight_list:
        rpn_model.load_weights(i, skip_mismatch=True, by_name=True)

    image_list = get_image_number_list()
    TP_list = [0] * len(Config.class_names)
    FP_list = [0] * len(Config.class_names)
    FN_list = [0] * len(Config.class_names)

    num_list = [[] for i in range(0, len(Config.class_names))]
    object_list = [0] * len(Config.class_names)
    ap_list = [0.0] * len(Config.class_names)

    def analysis_generator(img_list, show_image=False):
        while True:
            raw_image, raw_list = generate_single_image(img_list)
            image, shape = process_input_image(image=raw_image,
                                               input_dim=models.input_dim,
                                               method=method)
            image = np.expand_dims(image, axis=0)
            predictions = rpn_model.predict(image)
            res = rpn_output(predictions=predictions,
                             anchors=anchors,
                             top_k=models.rpn_result_batch,
                             confidence_threshold=0.5,
                             rpn_result_batch=models.rpn_result_batch,
                             iou_threshold=models.iou_threshold,
                             rpn_variacne=models.rpn_variance)[0]

            selected_area = pos2area(res[:, 1:], models.input_dim, models.rpn_stride)
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
                    if np.max(P_cls[0, ii, :]) < models.identifier_threshold or np.argmax(P_cls[0, ii, :]) == (
                            P_cls.shape[2] - 1):
                        continue
                    else:
                        box, conf, index = decode_classifier_result(model_name=model_name,
                                                                    cls=P_cls[0, ii, :],
                                                                    regr=P_regr[0, ii, :],
                                                                    roi=ROIs[0, ii, :],
                                                                    classifier_variance=models.classifier_variance)
                        box_list.append(box)
                        conf_list.append(conf)
                        index_list.append(index)
            index = np.array(index_list)
            conf = np.array(conf_list)
            box = np.array(box_list, dtype=np.float32)
            result_list = []
            if len(box) > 0:

                results = nms_for_out(all_labels=np.array(index),
                                      all_confs=np.array(conf),
                                      all_bboxes=np.array(box),
                                      num_classes=len(models.class_names),
                                      nms=models.identifier_threshold_nms,
                                      rpn_result_batch=models.rpn_result_batch)
                for item in range(0, len(results)):
                    result_list.append([results[item][2:], results[item][1], results[item][0]])
                if show_image:
                    draw_image_by_plt(image=raw_image,
                                      result_list=result_list,
                                      input_dim=models.input_dim,
                                      method=method)
            yield result_list, raw_list, raw_image

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
    for prediction_result, gt, raw_image in analysis_generator(image_list, show_image=False):
        img_shape = np.array(raw_image).shape
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
                tp = 0
                fp = 0
                total = len(k[k[:, 1] == 1])
                for j in range(0, len(k)):
                    if k[j, 1] > 0:
                        tp += 1
                    else:
                        fp += 1
                    prec.append(tp / (tp + fp))
                    rec.append(tp / total)
                r, g, b = random.random(), random.random(), random.random()
                plt.plot(rec, prec, linewidth=1, color=(r, g, b), marker='.', label="class {}".format(i))
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


def predict_for_rpn(rpn_weight_list, model_name, method, use_generator=True, use_annotation=False,
                    annotation_lines=None, run_on_laptop=False, use_bfc=True):
    models = model_name.value
    init_session(run_on_laptop, use_bfc)
    rpn_model, classifier_model = get_predict_model(model_name=model_name)

    for i in rpn_weight_list:
        rpn_model.load_weights(i, skip_mismatch=True, by_name=True)
    anchor = anchor_for_model(model_name=model_name)
    img_list = get_image_number_list()
    while True:
        if use_generator:
            raw_image, img_box = generate_single_image(img_list)
        elif use_annotation and annotation_lines is not None:
            line = annotation_lines.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            raw_image = Image.open(img_path)
        else:
            raise RuntimeError("No predict method selected.")
        image, shape = process_input_image(image=raw_image,
                                           input_dim=models.input_dim,
                                           method=method)
        image = np.expand_dims(image, axis=0)
        predictions = rpn_model.predict(image)
        result = rpn_output(predictions=predictions,
                            anchors=anchor,
                            top_k=models.rpn_result_batch,
                            rpn_result_batch=models.rpn_result_batch,
                            confidence_threshold=0.5,
                            iou_threshold=models.iou_threshold,
                            rpn_variacne=models.rpn_variance)[0]
        k = np.where(result[:, 0] > 0.5)
        r = []
        for item in k[0]:
            r.append([result[item][1:5], result[item][0], 0])
        draw_image_by_pillow(image=raw_image,
                             result_list=r,
                             input_dim=models.input_dim,
                             method=method,
                             show_label=False)


if __name__ == '__main__':
    # with open("data/nature_image.txt", "r", encoding="utf-8") as f:
    #     annotation_lines = f.readlines()
    # np.random.shuffle(annotation_lines)
    prediction_for_image(rpn_weight_list=[
        get_weight_file('rpn_ep082-loss562033.263-val_loss4.094-ModelConfig.VGG16-rpn.h5')],
        classifier_weight_list=[get_weight_file(
            'rpn_ep082-loss562033.263-val_loss4.094-ModelConfig.VGG16-rpn.h5')],
        model_name=ModelConfig.VGG16,
        method=ImageProcessMethod.Reshape,
        use_generator=True, run_on_laptop=False)
    # predict_for_rpn([get_weight_file('classifier_ep016-loss0.032-val_loss0.014-ModelConfig.ResNet50-classifier.h5')],
    #                 model_name=ModelConfig.ResNet50,
    #                 method=ImageProcessMethod.Reshape,
    #                 use_generator=True,
    #                 use_annotation=False)
    # prediction_for_recording(record_name="VGG16_generator", rpn_weight_list=[
    #     get_weight_file('rpn_ep016-loss0.233-val_loss0.153-ModelConfig.VGG16_standard-rpn.h5'), ],
    #                          classifier_model_weight=[get_weight_file(
    #                              'classifier_ep110-loss0.616-val_loss0.415-ModelConfig.VGG16-classifier.h5'), ],
    #                          model_name=ModelConfig.VGG16,
    #                          use_annotation=False,
    #                          use_generator=True,
    #                          generator_count=200)
    # prediction_for_analysis(
    #     [
    #         get_weight_file('rpn_ep016-loss0.233-val_loss0.153-ModelConfig.VGG16_standard-rpn.h5'), ],
    #     classifier_model_weight=[get_weight_file(
    #         'classifier_ep010-loss0.496-val_loss0.154-ModelConfig.VGG16-classifier.h5'), ],
    #     model_name=ModelConfig.VGG16,
    #     method=ImageProcessMethod.Reshape
    # )
    pass
