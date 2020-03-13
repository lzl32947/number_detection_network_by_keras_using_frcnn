import random

from PIL import Image
import numpy as np


def process_single_input(img, config):
    img = img.resize((600, 600), Image.ANTIALIAS)
    photo = np.array(img, dtype=np.float32)

    photo = process_pixel(photo)
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
