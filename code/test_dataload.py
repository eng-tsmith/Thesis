from preprocessing import batch_generator2, batch_generator_old, plot_image, l_c_r_data, flatten_data, center_val_data
from train import load_data, load_data_with_speed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt


class Args():
    def __init__(self, data_dir, test_size):
        self.data_dir = data_dir
        self.test_size = test_size


def plot_histogram(data_in):
    num_bins = 50

    avg_samples_per_bin = data_in.size / num_bins
    hist, bins = np.histogram(data_in, num_bins)

    width = 0.7 * (bins[1] - bins[0])
    cent = (bins[:-1] + bins[1:]) / 2
    plt.bar(cent, hist, align='center', width=width)
    plt.plot((np.min(data_in), np.max(data_in)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()
    return


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
    p1 = batch_generator2(data_dir, X_train, y_train, batch_size, True)
    p2 = batch_generator2(data_dir, X_valid, y_valid, batch_size, False)
else:
    # 2. Load data with speed
    X_train_speed, X_valid_speed, y_train_speed, y_valid_speed = load_data_with_speed(arg, print_enabled=False)
    p3 = batch_generator2(data_dir, X_train_speed, y_train_speed, batch_size, True)
    p4 = batch_generator2(data_dir, X_valid_speed, y_valid_speed, batch_size, False)


y_steer = y_train_speed[:,0]
y_speed = y_train_speed[:,1]

plot_histogram(y_steer)
plot_histogram(y_speed)

#for i in p4:
#    print(i[1][0])
