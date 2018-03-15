from preprocessing import batch_generator, flatten_data
import logging
from model import load_data
import cv2
from sklearn.model_selection import train_test_split
from preprocessing import process_img_for_visualization, denormalize_speed


class Args:
    def __init__(self, data_dir_in, test_size_in, batch_size_in, flatten_in, all_data_in, model_name_in, label_dim_in):
        self.data_dir = data_dir_in
        self.test_size = test_size_in
        self.batch_size = batch_size_in
        self.flatten = flatten_in
        self.all_data = all_data_in
        self.model_name = model_name_in
        self.label_dim = label_dim_in


def main(args_in):
    """
    Main funtion for testing of preprocessing
    :param args_in: Simulate argparser for hyperparameters
    """
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

    # Load Data
    x_train = None
    y_train = None

    try:
        logging.info('Loading data...')

        # Load data from CSV
        if args_in.all_data:
            x_data, y_data = load_data(args_in)
            x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data,
                                                                  test_size=args_in.test_size, random_state=5)
        else:
            logging.info('Loading train data')
            x_train, y_train = load_data(args_in, data_dir=data_dirs_train)
            logging.info('Loading validation data')
            x_valid, y_valid = load_data(args_in, data_dir=data_dirs_val)

        # Flatten distribution of steering angles
        if args_in.flatten:
            logging.info('Flatten data...')
            x_train, y_train = flatten_data(x_train, y_train, print_enabled=False, plot_enabled=False)
            x_valid, y_valid = flatten_data(x_valid, y_valid, print_enabled=False, plot_enabled=False)

        logging.info('Data loaded successfully')
        logging.info('Train on {} samples, validate on {} samples'.format(len(x_train), len(x_valid)))
    except Exception as e:
        logging.exception(e)
        logging.info('Data could not be loaded. Aborting')

    # Create generator
    p = batch_generator(args_in.data_dir, x_train, y_train, args_in.batch_size, args_in.label_dim,
                        True, args_in.model_name)

    cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

    # Run generator
    for i in p:
        for j in range(batch_size):
            im = i[0][j]
            tr_lenk = i[1][0][j]
            tr_velo = i[1][1][j]

            tr_velo = denormalize_speed(tr_velo)

            im = process_img_for_visualization(im, angle=tr_lenk)
            cv2.imshow('CNN input', im/255)
            logging.info("Steering Angle: {:.4f}  \tVelocity: {:.4f}".format(tr_lenk, tr_velo))
            cv2.waitKey(200)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ##############################
    # Test Pipeline
    ###############################
    # Setting
    data_dir = './rec_data'
    test_size = 0.2
    batch_size = 32
    flatten = True
    all_data = False
    model_name = 'nvidia'
    label_dim = 2

    # Create args like model.py
    args = Args(data_dir, test_size, batch_size, flatten, all_data, model_name, label_dim)

    main(args)
