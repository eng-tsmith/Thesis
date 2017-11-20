import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import logging

import argparse
import os
import time
import csv

from NN_arch.NVIDIA import build_model
from preprocessing import INPUT_SHAPE, batch_generator2, flatten_data, l_c_r_data, center_val_data

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)


def load_data(args):
    """
    Load training data from CSV and split it into training and validation set
    """
    #data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data = np.empty([1, 7])
    for subdir, dirs, files in os.walk(args.data_dir):
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

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    # Train data can be either of the center, left or right and the data is flattened to not prefer steering angles around 0
    X_train, y_train = l_c_r_data(X_train, y_train)
    X_train, y_train = flatten_data(X_train, y_train)

    # As the real data will always be the center image, validation will also only consist of middle image and does not have to be flattened
    X_valid = center_val_data(X_valid)

    logging.info('Train on {} samples, validate on {} samples'.format(len(X_train), len(X_valid)))

    return X_train, X_valid, y_train, y_valid


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    # Create directories for experiment
    dir_log = './logs/' + args.exp_name + '/'
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)

    # Save Hyperparameters
    with open(dir_log+'hyperparameters.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=':')
        csv_writer.writerow(['Experiment name', args.exp_name])
        csv_writer.writerow(['Learning rate', args.learning_rate])
        csv_writer.writerow(['Batch size', args.batch_size])
        csv_writer.writerow(['Number of epochs', args.nb_epoch])
        csv_writer.writerow(['Dropout probability', args.drop_prob])
        csv_writer.writerow(['Test size fraction', args.test_size])
        csv_writer.writerow(['Save best models only', args.save_best_only])

    # Callbacks
    checkpoint = ModelCheckpoint(dir_log + '/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    tensorboard = TensorBoard(log_dir=dir_log,
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=True)

    # Optimizer
    logging.info('Learning rate: ' + str(args.learning_rate))
    adam = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


    # Compile model
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae']) # mae = mean absoult error

    # Start Training
    model.fit_generator(batch_generator2(args.data_dir, X_train, y_train, args.batch_size, True),
                        steps_per_epoch=len(X_train)/args.batch_size,
                        epochs=args.nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint, tensorboard],
                        validation_data=batch_generator2(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid)/args.batch_size,
                        max_queue_size=1)

    logging.info("Finished Training")
    return


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
    # Params
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default=params[0])
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=float(params[1]))
    parser.add_argument('-k', help='drop out probability', dest='drop_prob', type=float, default=float(params[2]))
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=int(params[3]))
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=int(params[4]))
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default=params[5])
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=float(params[6]))
    parser.add_argument('-e', help='experiment name', dest='exp_name', type=str, default=params[7])
    args = parser.parse_args()

    if args.exp_name == 'Experiment Name':
        args.exp_name = str(time.time())

    logging.info('Experiment name: ' + args.exp_name)

    # print params
    logging.info('_' * 30)
    logging.info('Parameters')
    logging.info('=' * 30)
    for key, value in vars(args).items():
        logging.info('{:<20} := {}'.format(key, value))
        logging.info('_' * 30)

    # load data
    data = load_data(args)

    # build model
    start = time.time()
    model = build_model(args, INPUT_SHAPE)

    # train model on data, it saves as model.h5
    train_model(model, args, *data)

    elapsed = (time.time() - start)
    logging.info("The Training of the Network took" + str(elapsed) + " seconds to finish")


def main():
    """
       Load train/validation data set and train the model
    """
    # Logging
    logging.basicConfig(level=logging.INFO)
    # Parser
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='rec_data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='drop_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-e', help='experiment name', dest='exp_name', type=str, default=str(time.time()))
    args = parser.parse_args()

    logging.info('Experiment name: ' + args.exp_name)

    # print parameters
    logging.info('_' * 30)
    logging.info('Parameters')
    logging.info('=' * 30)
    for key, value in vars(args).items():
        logging.info('{:<20} := {}'.format(key, value))
        logging.info('_' * 30)

    # load data
    data = load_data(args)

    # build model
    start = time.time()
    model = build_model(args, INPUT_SHAPE)

    # train model on data, it saves as model.h5
    train_model(model, args, *data)

    elapsed = (time.time() - start)
    logging.info("The Training of the Network took" + str(elapsed) + " seconds to finish")


if __name__ == '__main__':
    main()
