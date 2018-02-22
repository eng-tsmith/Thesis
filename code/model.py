import os
import argparse
import time
import csv
import logging
from keras import backend as k_backend
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
# from NN_arch.NVIDIA import build_model
from NN_arch.ElectronGuy import build_model
from preprocessing import INPUT_SHAPE, flatten_data, batch_generator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Modify path for graphviz to work
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PATH"] += os.pathsep + 'D:/GraphViz/bin'

# Select GPU
if k_backend.backend() == 'tensorflow':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    k_backend.set_session(sess)
# ----------------------------------------------------------------------------


def divide_images(images_all, labels, angle_adj=0.2):
    """
    Reshape array and adjust angle of left and right images
    :param images_all: [center, left, right]
    :param labels: [steer, speed]
    :param angle_adj: offset for non center images
    :return: all images with corresponding steering angels
    """
    # images (n x 3) --> (3n x 1)
    x_out = np.reshape(images_all, images_all.shape[0] * images_all.shape[1], order='F')

    # labels
    # steering angel adjustment (x, x+adj, x-adj)
    y_1 = np.append(labels[:, 0], [labels[:, 0] + angle_adj, labels[:, 0] - angle_adj])
    # speed (does not change)
    y_2 = np.append(labels[:, 1], [labels[:, 1], labels[:, 1]])
    y_out = np.column_stack((y_1, y_2))

    return x_out, y_out


def load_data(args, data_dir=None):
    """
    Loads data from CSV file. Depending on data_dir it either loads all data in the root folder or only the ones defined
    in data_dir.
    :param args: arg parser
    :param data_dir: directory where CSVs are located. For all subdirectories data_dir = None
    :return: Returns the images(x_data) and labels(y_data)
    """
    df_all = pd.DataFrame(columns=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    if data_dir:
        for part in data_dir:
            part = os.path.join(args.data_dir, part)
            for subdir, dirs, files in os.walk(part):
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
                        df_new = pd.read_csv(os.path.join(os.getcwd(), os.path.join(subdir, file)),
                                             names=['center', 'left', 'right',
                                                    'steering', 'throttle', 'reverse', 'speed'])
                        df_all = df_all.append(df_new)

    else:
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
                    df_new = pd.read_csv(os.path.join(os.getcwd(), os.path.join(subdir, file)),
                                         names=['center', 'left', 'right',
                                                'steering', 'throttle', 'reverse', 'speed'])
                    df_all = df_all.append(df_new)

    images_all = df_all[['center', 'left', 'right']].values
    labels = df_all[['steering', 'speed']].values

    x_data, y_data = divide_images(images_all, labels)

    return x_data, y_data


def train_model(model, nn_name, args, x_train, x_valid, y_train, y_valid):
    """
    Function to train model. Saves all parameters to CSV file for every experiment. Measures time of training.
    :param model: built Keras model
    :param nn_name: name of neural net
    :param args: arg parser
    :param x_train: image for training
    :param x_valid: image for validation
    :param y_train: label for training
    :param y_valid: label for validation
    :return:
    """
    # Time measurements
    start = time.time()

    # Create directories for experiment
    dir_log = './logs/' + args.exp_name + '/'
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)

    # Save Hyperparameters
    with open(dir_log+'hyperparameters.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=':')
        csv_writer.writerow(['Experiment name: ', args.exp_name])
        csv_writer.writerow(['NN architecture: ', nn_name])
        csv_writer.writerow(['Learning rate: ', args.learning_rate])
        csv_writer.writerow(['Batch size: ', args.batch_size])
        csv_writer.writerow(['Number of epochs: ', args.nb_epoch])
        # csv_writer.writerow(['Dropout probability: ', args.drop_prob])
        csv_writer.writerow(['Test size fraction: ', args.test_size])
        csv_writer.writerow(['Save best models only: ', args.save_best_only])
        csv_writer.writerow(['Flatten steering angle data: ', args.flatten])
        csv_writer.writerow(['Label dimension: ', args.label_dim])

    # Callbacks
    checkpoint = ModelCheckpoint(dir_log + '/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    tensorboard = TensorBoard(log_dir=dir_log,
                              histogram_freq=0,
                              batch_size=args.batch_size,
                              write_graph=True,
                              write_grads=False,
                              write_images=True)

    # Optimizer
    adam = Adam(lr=args.learning_rate)  # ), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile model
    model.compile(loss='mse', optimizer=adam, metrics=['acc', 'mae'])

    # Plot model to PDF
    plot_model(model, to_file=dir_log + 'model_diagram.pdf', show_shapes=True, show_layer_names=True)

    # Start Training
    model.fit_generator(batch_generator(args.data_dir, x_train, y_train, args.batch_size, args.label_dim, True),
                        steps_per_epoch=len(x_train)/args.batch_size,
                        epochs=args.nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint, tensorboard],
                        validation_data=batch_generator(args.data_dir, x_valid, y_valid,
                                                        args.batch_size, args.label_dim, False),
                        validation_steps=len(x_valid)/args.batch_size,
                        max_queue_size=1)

    # Log duration of training
    elapsed = (time.time() - start)
    logging.info("Finished Training")
    logging.info("The Training of the Network took " + str(int(elapsed)) + " seconds to finish")

    with open(dir_log + 'hyperparameters.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=':')
        csv_writer.writerow(['Training duration', str(elapsed)])

    return


def s2b(s):
    """
    Converts a string to boolean value
    :param s: string
    :return: boolean
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Main function. Load train/validation data set, build model and train model
    :return:
    """
    # ----------------------------------------------------------------------------
    # Logging
    logging.basicConfig(level=logging.INFO)
    # ----------------------------------------------------------------------------
    # Parser
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-p', help='data directory', dest='data_dir', type=str, default='./rec_data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.05)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=512)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('-e', help='experiment name', dest='exp_name', type=str, default=str(time.time()))
    parser.add_argument('-s', help='predict speed', dest='pred_speed', type=s2b, default='true')
    parser.add_argument('-f', help='flatten data', dest='flatten', type=s2b, default='true')
    parser.add_argument('-d', help='label dimension', dest='label_dim', type=int, default='2')
    parser.add_argument('-a', help='use all data', dest='all_data', type=s2b, default='false')
    parser.add_argument('-q', help='dropout probability', dest='drop_prob', type=float, default=0.5)
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # Print parameters
    logging.info('=' * 30)
    logging.info('=' * 30)
    logging.info('Behavioral Cloning Training Program')
    logging.info('=' * 30)
    logging.info('=' * 30)
    logging.info('Experiment name: ' + args.exp_name)
    logging.info('_' * 30)
    logging.info('Parameters:')
    for key, value in vars(args).items():
        logging.info('{:<20} := {}'.format(key, value))
    logging.info('_' * 30)
    if not args.pred_speed:
        logging.info('Predicting only steering angle')
    else:
        logging.info('Predicting steering angle and speed')
    logging.info('_' * 30)
    logging.info('=' * 30)
    logging.info('=' * 30)
    # ----------------------------------------------------------------------------
    # Load data
    logging.info('Loading data...')

    # Manual train data definition
    data_dirs_train = [
        './berlin',
        './lake',
        './newyork',
        './recovery',
        './hongkong',
        './hongkong2',
        './montreal'
    ]
    # Manual val data definition
    data_dirs_val = [
        './jungle'
    ]

    try:
        if args.all_data:
            x_data, y_data = load_data(args)
            x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data,
                                                                  test_size=args.test_size, random_state=5)
        else:
            logging.info('Loading train data')
            x_train, y_train = load_data(args, data_dir=data_dirs_train)
            logging.info('Loading validation data')
            x_valid, y_valid = load_data(args, data_dir=data_dirs_val)

        # Flatten distribution of steering angles
        if args.flatten:
            logging.info('Flatten data...')
            x_train, y_train = flatten_data(x_train, y_train, print_enabled=False)
            x_valid, y_valid = flatten_data(x_valid, y_valid, print_enabled=False)

        logging.info('Data loaded successfully')
        logging.info('Train on {} samples, validate on {} samples'.format(len(x_train), len(x_valid)))
    except Exception as e:
        logging.exception(e)
        logging.info('Data could not be loaded. Aborting')
        return -1

    # ----------------------------------------------------------------------------
    # build model
    logging.info('Building model...')

    try:
        model, nn_name = build_model(args, INPUT_SHAPE)
        logging.info('Model built successfully')
    except Exception as e:
        logging.exception(e)
        logging.info('Model could not be built. Aborting')
        return -1

    # ----------------------------------------------------------------------------
    # Train model and save as model.h5
    logging.info('Training model...')
    try:
        train_model(model, nn_name, args, x_train, x_valid, y_train, y_valid)
        logging.info('Training finished')
    except Exception as e:
        logging.exception(e)
        logging.info('Training error. Aborting')
        return -1
    return 0


if __name__ == '__main__':
    main()
