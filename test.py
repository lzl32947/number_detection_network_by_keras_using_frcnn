from keras_applications import imagenet_utils

from config.configs import Config
from network.backbone.resnet50 import ResNet50
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.layers import *
import numpy as np
from PIL import Image
from network.rpn.common_rpn import get_rpn
import matplotlib.pyplot as plt
from util.data_util import get_data, get_random_data, generate
from util.decode_util import nms_for_out, rpn_output
from util.draw_util import draw_result


def get_img_output_length(width, height):
    def get_output_length(input_length):
        input_length += 6
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


if __name__ == '__main__':
    annotation_path = '2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    # for i in lines:
    #     k = get_random_data(i)
    #     fig = plt.figure()
    #     plt.imshow(k[0].astype(np.uint8))
    #     ax = fig.add_subplot(1, 1, 1)
    #     for j in k[1]:
    #         rect = plt.Rectangle((j[0], j[1]), j[2] - j[0], j[3] - j[1], fill=False, edgecolor='red', linewidth=1)
    #         ax.add_patch(rect)
    #     plt.show()
    #     plt.close()
    rpn_train = generate(lines, 11, data_function=get_data)
    rpn_train_2 = generate(lines, 11, data_function=get_random_data)
    for i, j in zip(rpn_train, rpn_train_2):
        print(i[2][0][:, 4])
