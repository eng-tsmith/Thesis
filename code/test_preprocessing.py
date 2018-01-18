from preprocessing import batch_generator2, batch_generator_old, plot_image
import logging
logging.basicConfig(level=logging.INFO)
from train import load_data, load_data_with_speed
import cv2
import numpy as np


class Args():
    def __init__(self, data_dir, test_size):
        self.data_dir = data_dir
        self.test_size = test_size

##############################
# Test Pipeline
###############################
##############################
# Test Pipeline
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 32
arg = Args(data_dir, test_size)

test_speed = True

# 1. Load Data
if not test_speed:
    X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=False)
    p = batch_generator2(data_dir, X_train, y_train, batch_size, False)
else:
    X_train, X_valid, y_train, y_valid = load_data_with_speed(arg, print_enabled=False)
    p = batch_generator2(data_dir, X_train, y_train, batch_size, False)

cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

for i in p:
    image = i[0][0]
    train = i[1][0]

    cv2.imshow('CNN input', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(train)
    cv2.waitKey(200)


