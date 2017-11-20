import preprocessing
from preprocessing import flatten_data, l_c_r_data, load_image, center_val_data, batch_generator2, plot_image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from random import randint
import logging
logging.basicConfig(level=logging.INFO)

batch_size = 1
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
data_dir = './rec_data'
test_size = 0.2
images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
steers = np.empty(batch_size)


def load_data(data_dir, test_size):
    """
    Load training data from CSV and split it into training and validation set
    """
    # data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data = np.empty([1, 7])
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file == 'driving_log.csv':
                logging.info("Loading : " + os.path.join(subdir, file))
                try:
                    if dirs[0] != 'IMG':
                        logging.info("Missing IMG directory in " + subdir)
                        break
                except(IndexError):
                    logging.info("No directories!")
                    break
                data_df = pd.read_csv(os.path.join(os.getcwd(), os.path.join(subdir, file)),
                                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
                data = np.append(data, data_df.values, axis=0)
    data = np.delete(data, (0), axis=0)

    # names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    X = data[:, 0:3]
    y = data[:, 3]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)

    # Train data can be either of the center, left or right and the data is flattened to not prefer steering angles around 0
    X_train, y_train = l_c_r_data(X_train, y_train)
    X_train, y_train = flatten_data(X_train, y_train)

    # As the real data will always be the center image, validation will also only consist of middle image and does not have to be flattened
    X_valid = center_val_data(X_valid)

    logging.info('Train on {} samples, validate on {} samples'.format(len(X_train), len(X_valid)))

    return X_train, X_valid, y_train, y_valid


###############################
# Test Pipeline
###############################

# 1. Load Data
X_train, X_valid, y_train, y_valid = load_data(data_dir, test_size)
p = batch_generator2(data_dir, X_train, y_train, batch_size, True)

for i in p:
    image = i[0][0]
    train = i[1]

    print(train)
    plot_image(image)
