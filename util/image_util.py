import os
import random

from PIL.ImageDraw import Draw
import matplotlib.pyplot as plt
from config.Configs import Config, ImageProcessMethod
from PIL import Image, ImageFont
import numpy as np

from util.output_util import process_output_image


def resize_image(image, input_dim):
    """
    Return the resized image.
    :param input_dim: int, the input dimension
    :param image: PIL.Image object
    :return: the resized new image
    """
    new_img = image.resize((input_dim, input_dim), Image.ANTIALIAS)
    return new_img


def zoom_image(image, input_dim):
    """
    Scaled the image to the center.
    :param input_dim: int, the input dimension
    :param image: PIL Image object
    :return: the scaled image
    """
    img_shape = np.array(np.shape(image)[0:2])
    width = img_shape[1]
    height = img_shape[0]
    width_zoom_ratio = input_dim / width
    height_zoom_ratio = input_dim / height
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * width)
    new_height = int(zoom_ratio * height)
    img_t = image.resize((new_width, new_height), Image.BICUBIC)
    new_img = Image.new('RGB', (input_dim, input_dim), (128, 128, 128))

    width_offset = (input_dim - new_width) // 2
    height_offset = (input_dim - new_height) // 2
    new_img.paste(img_t, (width_offset, height_offset))
    return new_img


def draw_image_by_plt(image, input_dim, result_list, method, show_label=True, show_conf=True, print_result=True,
                      is_rpn=False):
    """
    Draw the image with matplotlab.pyplot and show it.
    :param is_rpn: whether this image is for rpn showing.
    :param input_dim: int, the input dimension
    :param print_result: bool, whether to print the identification result to console
    :param image: PIL.Image object
    :param result_list: list, the list of the result, in format of (box, conf, index)
    :param method: PMethod class, the method to process the image
    :param show_label: bool, whether to show the label
    :param show_conf: bool, whether to show the confidence
    :return: the image or None, defined in return_image
    """
    plt.figure()
    plt.imshow(np.array(image, dtype=np.uint8))

    shape = np.array(image).shape
    width_zoom_ratio = input_dim / shape[1]
    height_zoom_ratio = input_dim / shape[0]
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * shape[1])
    new_height = int(zoom_ratio * shape[0])

    width_offset = (input_dim - new_width) // 2
    height_offset = (input_dim - new_height) // 2
    width_offset /= input_dim
    height_offset /= input_dim

    print("identification result:")
    for box, conf, index in result_list:
        x_min, y_min, x_max, y_max = process_output_image(
            method=method,
            input_dim=input_dim,
            original_shape=shape,
            original_box=box,
            width_offset=width_offset,
            height_offset=height_offset
        )
        r, g, b = random.random(), random.random(), random.random()
        plt.gca().add_patch(
            plt.Rectangle((x_min, y_min), x_max - x_min,
                          y_max - y_min, fill=False,
                          edgecolor=(r, g, b), linewidth=1)
        )
        if show_conf and show_label:
            plt.text(x_min + 2, y_min + 2, "{:.2f}% \n{}".format(conf * 100, int(index)))
        else:
            if show_label:
                plt.text(x_min + 2, y_min + 2, "{}".format(int(index)))
            if show_conf:
                plt.text(x_min + 2, y_min + 2, "{:.2f}%".format(conf * 100))
        if print_result:
            if is_rpn:
                print("location:{},{},{},{}\tconf:{:.2f}%".format(int(index), int(x_min), int(y_min), int(x_max),
                                                                  int(y_max), conf * 100))
            else:
                print("class:{}\tlocation:{},{},{},{}\tconf:{:.2f}%".format(int(index), int(x_min), int(y_min),
                                                                            int(x_max),
                                                                            int(y_max), conf * 100))
    print("identification result end.")
    plt.show()
    plt.close()


def draw_image_by_pillow(image, input_dim, result_list, method, show_label=True, show_conf=True, return_image=False,
                         print_result=True, is_rpn=False):
    """
    Draw the image and show it.
    :param is_rpn: whether this image is for rpn showing.
    :param input_dim: int, the input dimension
    :param print_result: bool, whether to print the identification result to console
    :param image: PIL.Image object
    :param result_list: list, the list of the result, in format of (box, conf, index)
    :param method: PMethod class, the method to process the image
    :param show_label: bool, whether to show the label
    :param show_conf: bool, whether to show the confidence
    :param return_image: bool, whether to return the image
    :return: the image or None, defined in return_image
    """
    shape = np.array(image).shape

    font_size = max(shape[0] // 20, 10)
    font = ImageFont.truetype(os.path.join(Config.font_dir, "simhei.ttf"), font_size)

    draw = Draw(image)
    width_zoom_ratio = input_dim / shape[1]
    height_zoom_ratio = input_dim / shape[0]
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * shape[1])
    new_height = int(zoom_ratio * shape[0])

    width_offset = (input_dim - new_width) // 2
    height_offset = (input_dim - new_height) // 2
    width_offset /= input_dim
    height_offset /= input_dim
    print("identification result:")
    for box, conf, index in result_list:
        x_min, y_min, x_max, y_max = process_output_image(
            method=method,
            input_dim=input_dim,
            original_shape=shape,
            original_box=box,
            width_offset=width_offset,
            height_offset=height_offset
        )
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(r, g, b))
        if show_conf and show_label:
            draw.text((x_min + 2, y_min + 2), "{:.2f}% \n{}".format(conf * 100, int(index)), font=font, fill=(r, g, b))
        else:
            if show_label:
                draw.text((x_min + 2, y_min + 2), "{}".format(int(index)), font=font,
                          fill=(r, g, b))
            if show_conf:
                draw.text((x_min + 2, y_min + 2), "{:.2f}%".format(conf * 100), font=font,
                          fill=(r, g, b))
        if print_result:
            if is_rpn:
                print("location:{},{},{},{}\tconf:{:.2f}%".format(int(index), int(x_min), int(y_min), int(x_max),
                                                                  int(y_max), conf * 100))
            else:
                print("class:{}\tlocation:{},{},{},{}\tconf:{:.2f}%".format(int(index), int(x_min), int(y_min),
                                                                            int(x_max),
                                                                            int(y_max), conf * 100))
    print("identification result end.")
    if not return_image:
        image.show()
    else:
        return image


def plot_classifier_train_data(x, y, feature_map_size):
    """
    This function plot the ROIs on to the input image, used for plot x,y in classifier date generator.
    :param x: list, data of classifier data generator
    :param y: list, data of classifier data generator
    :param feature_map_size: int, the size of feature map
    :return: None
    """
    image = np.squeeze(x['image'])
    roi_list = x['roi']
    label = y['classification_1']
    regression_data = y['regression_1']

    def de_process_image(img_array):
        mean = [103.939, 116.779, 123.68]
        img_array[..., 0] += mean[0]
        img_array[..., 1] += mean[1]
        img_array[..., 2] += mean[2]
        img_array = img_array[..., ::-1]
        return np.array(img_array, dtype=np.uint8)

    image = de_process_image(image)
    plt.figure()
    plt.imshow(image)
    input_shape = image.shape[0:2]
    ratio_x = input_shape[0] / feature_map_size
    ratio_y = input_shape[1] / feature_map_size
    for i in range(0, len(roi_list)):
        for item in range(0, len(roi_list[i])):
            p = roi_list[i, item]
            label_index = np.argmax(label[i, item])
            if label_index == len(Config.class_names):
                continue
            x, y, w, h = round(p[0] * ratio_x), round(p[1] * ratio_y), round(p[2] * ratio_x), round(p[3] * ratio_y)
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, fill=False,
                              edgecolor='r', linewidth=1)
            )

            plt.text(p[0] * ratio_x + 2, p[1] * ratio_y + 2, "{}".format(int(label_index)))
    plt.show()
    plt.close()
