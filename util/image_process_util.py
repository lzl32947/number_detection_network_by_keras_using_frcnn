import random

from PIL import Image
import numpy as np

from config.configs import Config


def process_single_input(img):
    # Just resize the image to 600,600
    img = img.resize((Config.input_dim, Config.input_dim), Image.ANTIALIAS)
    # change the dtype and then for the pixel process
    photo = np.array(img, dtype=np.float32)
    photo = process_pixel(photo)
    # expand the dim to make it has four dimensions
    photo = np.expand_dims(photo, axis=0)

    return photo, img


def process_pixel(img_array):
    # 该函数旨在简化keras中preprocess_input函数的工作过程
    # 和preprocess_input函数返回相同的值
    # 'RGB'->'BGR'
    img_array = img_array[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array
