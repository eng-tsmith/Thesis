from preprocessing import batch_generator2, batch_generator_old, plot_image, l_c_r_data, flatten_data, center_val_data
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

def load_data(args, print_enabled=False):
    """
    Load training data from CSV and split it into training and validation set
    """
    data_list = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file == 'driving_log.csv':
                logging.info("Loading : " + os.path.join(subdir, file))
                try:
                    if dirs[0] != 'IMG':
                        logging.info("Missing IMG directory in " + subdir)
                        break
                except IndexError:
                    logging.info("No directories!")
                    break
                data_df = pd.read_csv(os.path.join(os.getcwd(), os.path.join(subdir, file)),
                                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
                data_list.append(data_df)

    data = pd.concat(data_list)
    X = data[['center', 'left', 'right']].values
    y = data['steering'].values

    # Split dataset to train and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    # Divide into single images (center, left and right) and flatten train data (get rid of too many steering angles 0.0)
    X_train, y_train = l_c_r_data(X_train, y_train)
    X_train, y_train = flatten_data(X_train, y_train, print_enabled=print_enabled)

    # Only center image for validation as this is also done in real-time mode
    X_valid = center_val_data(X_valid)

    # Print size of dataset
    logging.info('Train on {} samples, validate on {} samples'.format(len(y_train), len(y_valid)))

    return X_train, X_valid, y_train, y_valid


##############################
# Test Pipeline
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 32

# 1. Load Data
arg = Args(data_dir, test_size)
X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=False)
p = batch_generator2(data_dir, X_train, y_train, batch_size, False)

print(X_train[0])


print(os.path.join(arg.data_dir, X_train[0].split("\\")[-3], X_train[0].split("\\")[-2], X_train[0].split("\\")[-1]))