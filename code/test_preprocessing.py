from preprocessing import batch_generator, flatten_data
import logging
from model import load_data
import cv2
from sklearn.model_selection import train_test_split
from preprocessing import plot_image, process_img_for_visualization
import numpy as np

logging.basicConfig(level=logging.INFO)


class Args:
    def __init__(self, data_dir, test_size, flatten, all_data):
        self.data_dir = data_dir
        self.test_size = test_size
        self.flatten = flatten
        self.all_data = all_data




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
all_data = False

args = Args(data_dir, test_size, flatten, all_data)

test_speed = True

# Manual train data definition
data_dirs_train = [
    './berlin',
    './berlin2'
    './berlin3',
    './hongkong',
    './hongkong2',
    './hongkong3',
    './jungle',
    './jungle2',
    './lake',
    './lake2',
    './montreal',
    './montreal2',
    './newyok2',
    './newyork',
    './newyork3'
]
# Manual val data definition
data_dirs_val = [
'./lake'
]

# 1. Load Data
try:
    logging.info('Loading data...')

    # Load data from CSV
    if args.all_data:
        x_data, y_data = load_data(args)
        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=args.test_size, random_state=5)
    else:
        logging.info('Loading train data')
        x_train, y_train = load_data(args, data_dir=data_dirs_train)
        logging.info('Loading validation data')
        x_valid, y_valid = load_data(args, data_dir=data_dirs_val)

    # Flatten distribution of steering angles
    if args.flatten:
        logging.info('Flatten data...')
        x_train, y_train = flatten_data(x_train, y_train, print_enabled=False, plot_enabled=False)
        x_valid, y_valid = flatten_data(x_valid, y_valid, print_enabled=False, plot_enabled=False)

    logging.info('Data loaded successfully')
    logging.info('Train on {} samples, validate on {} samples'.format(len(x_train), len(x_valid)))
except Exception as e:
    logging.exception(e)
    logging.info('Data could not be loaded. Aborting')


if not test_speed:
    label_dim = 1
    p = batch_generator(data_dir, x_train, y_train, batch_size, label_dim, False)
else:
    label_dim = 2
    p = batch_generator(data_dir, x_train, y_train, batch_size, label_dim, True)

cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

for i in p:
    image = i[0][0]
    train = i[1][0]

    for j in range(batch_size):
        im = i[0][j]
        tr = i[1][j]

        im = process_img_for_visualization(im, angle=tr[0])

        cv2.imshow('CNN input', im/255)
        print(tr)
        cv2.waitKey(200)



    # plot_image(image)
    # print(i[0].shape)
    # print(i[1].shape)

    # im = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    # im = process_img_for_visualization(image, angle=i[1][0][0])
    #
    # cv2.imshow('CNN input', im/255)
    # print(train)
    # cv2.waitKey(200)


