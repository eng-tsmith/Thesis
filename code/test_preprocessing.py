import preprocessing
from preprocessing import flatten_data, l_c_r_data, load_image, center_val_data
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
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train data can be either of the center, left or right and the data is flattened to not prefer steering angles around 0
    X_train, y_train = l_c_r_data(X_train, y_train)
    X_train, y_train = flatten_data(X_train, y_train, print_enabled=True)

    # As the real data will always be the center image, validation will also only consist of middle image and does not have to be flattened
    X_valid = center_val_data(X_valid)

    return X_train, X_valid, y_train, y_valid


def plot_image(image_display):
    plt.imshow(image_display)
    plt.show()

###############################
# Test Pipeline
###############################

# 1. Load Data
X_train, X_valid, y_train, y_valid = load_data(data_dir, test_size)

# 2. Choose random sample within dataset
index = randint(0, y_train.size)
img = X_train[index]
#center, left, right = X_train[index]
steering_angle = y_train[index]

# 3. Dataaugmentation steps
"""
Generate an augumented image and adjust steering angle.
(The steering angle is associated with the center image)
"""
image = load_image(data_dir, img)
plot_image(image)

image, steering_angle = preprocessing.random_flip(image, steering_angle)
plot_image(image)

image, steering_angle = preprocessing.random_translate(image, steering_angle, 100, 10)
plot_image(image)

image = preprocessing.random_shadow(image)
plot_image(image)

image = preprocessing.random_brightness(image)
plot_image(image)

# 4. Preprocessing
image = preprocessing.crop(image)
plot_image(image)

image = preprocessing.resize(image)
plot_image(image)

image = preprocessing.rgb2yuv(image)
plot_image(image)
