# Number detection network using Faster R-CNN

Author: Liu Zhilin      Lanzhou University

This repo implement one number detection network with the structure of Faster R-CNN

* Dependence
```
    keras > 2.0 
    1.13 < tensorflow < 2.0
```

* Project structure
```
root
├─config
│   Configs.py          # Contain config of whole project
│
├─data                  # Contain data of project
|  └─single_digits      # Contain single digit
├─log
│  ├─checkpoint         # Store checkpoint when train
│  ├─tensorboard        # Store tensorboard record when train
|  └─weight             # Contain the pre-trained weights
├─model
│  ├─image              # Store the image of model
│  └─layers             # Contain the custom layers of SSD
│  *.py                 # The concrete model
├─other
│  └─font               # Contain the font py drawing
├─util                  # Contain the function that process the model/input/output
│   image_generator.py  # The image generator functions
│   image_util.py       # The drawing/resizing functions
│   input_util.py       # The data pre-process functions before training
│   output_util.py      # The data process functions that decode the output from the model
├─train.py              # Function of training the model.
└─predcit.py            # Function of predict on model.

```