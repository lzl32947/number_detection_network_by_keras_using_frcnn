import os
import random

from PIL.ImageDraw import Draw

from config.Configs import Config, PMethod
from PIL import Image, ImageFont
import numpy as np


def resize_image(image):
    """
    Return the resized image.
    :param image: PIL.Image object
    :return: the resized new image
    """
    new_img = image.resize((Config.input_dim, Config.input_dim), Image.ANTIALIAS)
    return new_img


def zoom_image(image):
    """
    Scaled the image to the center.
    :param image: PIL Image object
    :return: the scaled image
    """
    img_shape = np.array(np.shape(image)[0:2])
    width = img_shape[1]
    height = img_shape[0]
    width_zoom_ratio = Config.input_dim / width
    height_zoom_ratio = Config.input_dim / height
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * width)
    new_height = int(zoom_ratio * height)
    img_t = image.resize((new_width, new_height), Image.BICUBIC)
    new_img = Image.new('RGB', (Config.input_dim, Config.input_dim), (128, 128, 128))

    width_offset = (Config.input_dim - new_width) // 2
    height_offset = (Config.input_dim - new_height) // 2
    new_img.paste(img_t, (width_offset, height_offset))
    return new_img


def draw_image(image_path, result_list, method, show_label=True, show_conf=True, return_image=False):
    """
    Draw the image and show it.
    :param image_path: str, the original image
    :param result_list: list, the list of the result, in format of (box, conf, index)
    :param method: PMethod class, the method to process the image
    :param show_label: bool, whether to show the label
    :param show_conf: bool, whether to show the confidence
    :param return_image: bool, whether to return the image
    :return: the image or None, defined in return_image
    """
    image = Image.open(image_path)
    shape = np.array(image).shape

    font_size = max(shape[0] // 20, 5)
    font = ImageFont.truetype(os.path.join(Config.font_dir, "simhei.ttf"), font_size)

    draw = Draw(image)
    width_zoom_ratio = Config.input_dim / shape[1]
    height_zoom_ratio = Config.input_dim / shape[0]
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * shape[1])
    new_height = int(zoom_ratio * shape[0])

    width_offset = (Config.input_dim - new_width) // 2
    height_offset = (Config.input_dim - new_height) // 2
    width_offset /= Config.input_dim
    height_offset /= Config.input_dim

    for box, conf, index in result_list:
        if method == PMethod.Reshape:
            x_min = shape[1] * box[0]
            y_min = shape[0] * box[1]
            x_max = shape[1] * box[2]
            y_max = shape[0] * box[3]
        elif method == PMethod.Zoom:
            x_min = Config.input_dim * box[0]
            y_min = Config.input_dim * box[1]
            x_max = Config.input_dim * box[2]
            y_max = Config.input_dim * box[3]
            if width_offset > 0:
                if x_min < Config.input_dim * 0.5:
                    x_min = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - x_min) * shape[0] / shape[1])
                else:
                    x_min = Config.input_dim * 0.5 + (x_min - Config.input_dim * 0.5) * shape[0] / shape[1]
                if x_max < Config.input_dim * 0.5:
                    x_max = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - x_max) * shape[0] / shape[1])
                else:
                    x_max = Config.input_dim * 0.5 + (x_max - Config.input_dim * 0.5) * shape[0] / shape[1]
            if height_offset > 0:
                if y_min < Config.input_dim * 0.5:
                    y_min = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - y_min) * shape[1] / shape[0])
                else:
                    y_min = Config.input_dim * 0.5 + (y_min - Config.input_dim * 0.5) * shape[1] / shape[0]
                if y_max < Config.input_dim * 0.5:
                    y_max = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - y_max) * shape[1] / shape[0])
                else:
                    y_max = Config.input_dim * 0.5 + (y_max - Config.input_dim * 0.5) * shape[1] / shape[0]
            x_min = x_min / Config.input_dim * shape[1]
            x_max = x_max / Config.input_dim * shape[1]
            y_min = y_min / Config.input_dim * shape[0]
            y_max = y_max / Config.input_dim * shape[0]
        else:
            raise RuntimeError("No Method Selected.")
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
    if not return_image:
        image.show()
    else:
        return image
