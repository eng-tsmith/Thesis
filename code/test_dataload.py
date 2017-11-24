from preprocessing import batch_generator2, batch_generator_old, plot_image, l_c_r_data, flatten_data, center_val_data
from train import load_data, load_data_with_speed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
logging.basicConfig(level=logging.INFO)



class Args():
    def __init__(self, data_dir, test_size):
        self.data_dir = data_dir
        self.test_size = test_size


##############################
# Test Pipeline
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 32

# 1. Load Data
arg = Args(data_dir, test_size)
X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=False)
p1 = batch_generator2(data_dir, X_train, y_train, batch_size, True)
p2 = batch_generator2(data_dir, X_valid, y_valid, batch_size, False)

# 2. Load data with speed
X_train_speed, X_valid_speed, y_train_speed, y_valid_speed = load_data_with_speed(arg, print_enabled=False)
p3 = batch_generator2(data_dir, X_train_speed, y_train_speed, batch_size, True)
p4 = batch_generator2(data_dir, X_valid_speed, y_valid_speed, batch_size, False)

for i in p4:
    print(i[1][0])