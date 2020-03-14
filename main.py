import argparse
import os

from config.configs import Config
from predict import predict_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predict", action="store_true", help="Run the program to predict images")
    parser.add_argument("-i", "--images", help="Predict images in provided path.",
                        default="G:\\data_stored\\generated_train")
    parser.add_argument("-m", "--model", help="The model or the weight path.",
                        default="weight/number_weight.h5")
    args = parser.parse_args()

    config = Config()
    config.model_path = args.model
    if args.predict:
        if os.path.isdir(args.images):
            l = []
            for root, dirs, files in os.walk(args.images):
                for item in files:
                    l.append(os.path.join(root, item))
        else:
            l = [args.images, ]
        predict_images(l, config)
