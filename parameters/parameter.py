import colorsys


class HyperParameters(object):
    min_dim = 600


class NormalParameters(object):

    def __init__(self) -> None:
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), self.hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


