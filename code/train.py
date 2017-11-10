import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils import INPUT_SHAPE, batch_generator

import argparse
import os

from models.NVIDIA import build_model

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)

# Global Variables
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=True)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        steps_per_epoch=args.samples_per_epoch/args.batch_size,
                        epochs=args.nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint, tensorboard],
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid)/args.batch_size,
                        max_queue_size=1
                        )
#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def run(params):
    """
       Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default=params[0])
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=float(params[1]))
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=float(params[2]))
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=int(params[3]))
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=int(params[4]))
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=int(params[5]))
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default=params[6])
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=float(params[7]))
    args = parser.parse_args()

    global INPUT_SHAPE

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)


    #load data
    data = load_data(args)

    #build model
    model = build_model(args, INPUT_SHAPE)

    #train model on data, it saves as model.h5
    train_model(model, args, *data)


def main():
    """
       Load train/validation data set and train the model
       """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='rec_data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    # print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # load data
    data = load_data(args)
    # build model
    model = build_model(args)
    # train model on data, it saves as model.h5
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
