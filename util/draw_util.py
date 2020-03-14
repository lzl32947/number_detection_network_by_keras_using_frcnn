import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.configs import Config


def random_color():
    """
    get random color.
    :return: str: like "#112233"
    """
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def draw_result(image, results):
    top_label_indices = results[:, 0]
    top_conf = results[:, 1]
    boxes = results[:, 2:]
    # change it to real_size
    boxes[:, 0] = boxes[:, 0] * 600
    boxes[:, 1] = boxes[:, 1] * 600
    boxes[:, 2] = boxes[:, 2] * 600
    boxes[:, 3] = boxes[:, 3] * 600
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

    thickness = (np.shape(image)[0] + np.shape(image)[1]) // 600
    for i, c in enumerate(top_label_indices):
        predicted_class = Config.class_names[int(c)]
        score = top_conf[i]

        left, top, right, bottom = boxes[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline='blue')
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill='red')
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    image.show()
