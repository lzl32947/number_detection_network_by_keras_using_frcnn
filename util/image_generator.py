import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
import random
from PIL import Image
import os

from config.Configs import Config


def rect_cross(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    """
    This function calculate whether the two rectangle cross over each other.
    :return: the two rectangles have inner set
    """
    zx = abs(x1_min + x1_max - x2_min - x2_max)
    x = abs(x1_min - x1_max) + abs(x2_min - x2_max)
    zy = abs(y1_min + y1_max - y2_min - y2_max)
    y = abs(y1_min - y1_max) + abs(y2_min - y2_max)
    if zx <= x and zy <= y:
        return 1
    else:
        return 0


def get_image_number_list():
    """
    gather the image of each number to a list
    :return: the list of files
    """
    f_list = []
    for class_name in Config.class_names:
        single_digit_list = []
        for root, dirs, files in os.walk(os.path.join(Config.single_digits_dir, class_name)):
            for file in files:
                full_path = os.path.join(root, file)
                single_digit_list.append(full_path)
        f_list.append(single_digit_list)
    return f_list


def generate_single_image(single_image_list):
    """
    Return the generated image and the ground truth box.
    :param single_image_list: list, the file path to single image
    :return: the image array, the box
    """
    while True:
        number_list = []
        image_height = random.randint(40, 60)
        image_width = random.randint(image_height * 4, image_height * 8)

        new_img = np.random.rand(image_height, image_width, 3)
        new_img = new_img * 255
        new_img = new_img.astype(np.uint8)
        new_img = Image.fromarray(new_img)

        for i in range(0, 8):
            pick_num = random.randint(0, len(Config.class_names) - 1)
            index = random.randint(0, len(single_image_list[pick_num]) - 1)
            number_image = Image.open(single_image_list[pick_num][index])
            width = np.shape(number_image)[1]
            height = np.shape(number_image)[0]
            if len(number_list) > 0:
                try_time = 10
                while try_time > 0:
                    x_min = random.randint(0, image_width - width - 1)
                    y_min = random.randint(0, image_height - height - 1)
                    x_max = x_min + width
                    y_max = y_min + height
                    flag_conflict = False
                    for item_dict in number_list:
                        if rect_cross(x_min, y_min, x_max, y_max, item_dict[0], item_dict[1],
                                      item_dict[2], item_dict[3]):
                            flag_conflict = True
                            break
                    if not flag_conflict:
                        number_dict = [
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                            pick_num
                        ]
                        new_img.paste(number_image, (x_min, y_min))
                        number_list.append(number_dict)
                        break
                    else:
                        try_time -= 1

                if try_time == 0:
                    continue

            else:
                x_min = random.randint(0, image_width - width - 1)
                y_min = random.randint(0, image_height - height - 1)
                x_max = x_min + width
                y_max = y_min + height
                number_dict = [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    pick_num
                ]
                new_img.paste(number_image, (x_min, y_min))
                number_list.append(number_dict)
        break
    return new_img, np.array(number_list)
