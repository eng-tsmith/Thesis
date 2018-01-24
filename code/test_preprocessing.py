from preprocessing import batch_generator
import logging
logging.basicConfig(level=logging.INFO)
from model import load_data
import cv2
import numpy as np


class Args():
    def __init__(self, data_dir, test_size, flatten):
        self.data_dir = data_dir
        self.test_size = test_size
        self.flatten = flatten

##############################
# Test Pipeline
###############################
##############################
# Test Pipeline
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 32
flatten = True
arg = Args(data_dir, test_size, flatten)

test_speed = True

# 1. Load Data
if not test_speed:
    label_dim = 1
    X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=False)
    p = batch_generator(data_dir, X_train, y_train, batch_size, label_dim, False)
else:
    label_dim = 2
    X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=True)
    p = batch_generator(data_dir, X_train, y_train, batch_size, label_dim, True)

cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

for i in p:
    image = i[0][0]
    train = i[1][0]

    cv2.imshow('CNN input', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(train)
    cv2.waitKey(200)


