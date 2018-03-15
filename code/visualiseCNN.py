from keras.models import load_model
from preprocessing import load_image_absolute, preprocess
import numpy as np
import time
from keras import backend as k_backend
from scipy.misc import imsave
import logging
import argparse
# Manual deactivation of learning mode for backend functions
k_backend.set_learning_phase(0)


def s2b(s):
    """
    Converts a string to boolean value
    :param s: string
    :return: boolean
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def normalize(x):
    """
    utility function to normalize a tensor by its L2 norm
    :param x: Input
    :return: Normalized Input
    """

    return x / (k_backend.sqrt(k_backend.mean(k_backend.square(x))) + k_backend.epsilon())


def deprocess_image(x):
    """
    Denormalize Input
    :param x: Input
    :return: Denormalized Input
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def main(args_in):
    """
    Main funtion to visualise CONV filters
    Source: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :return:
    """
    # Compute real image
    path_image = args_in.image_path
    use_real_img = args_in.use_real_img

    # build model
    path_weights = args_in.model_path
    model = load_model(path_weights)
    model.summary()

    if model.input_shape[1] == 66 and model.input_shape[2] == 200:
        logging.info('nvidia')
        model_name = 'nvidia'
        # dimensions of the generated pictures for each filter.
        img_height = 66
        img_width = 200

        layers = ['conv1',
                  'conv2',
                  'conv3',
                  'conv4',
                  'conv5'
                  ]

        nr_filter_dict = {
            layers[0]: 24,
            layers[1]: 36,
            layers[2]: 48,
            layers[3]: 64,
            layers[4]: 64,
        }
    elif model.input_shape[1] == 64 and model.input_shape[2] == 64:
        logging.info('electron')
        model_name = 'electron'
        # dimensions of the generated pictures for each filter.
        img_height = 64
        img_width = 64
        layers = ['conv_layer_1',
                  'conv_layer_2',
                  'conv_layer_3',
                  'conv_layer_4',
                  'conv_layer_5',
                  'conv_layer_6',
                  'conv_layer_7',
                  'conv_layer_8'
                  ]

        nr_filter_dict = {
            layers[0]: 16,
            layers[1]: 16,
            layers[2]: 32,
            layers[3]: 32,
            layers[4]: 64,
            layers[5]: 64,
            layers[6]: 128,
            layers[7]: 128,
        }
    else:
        logging.info('Model not found')
        return -1

    for curr_layer_name in layers:
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # this is the placeholder for the input images
        input_img = model.input

        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        kept_filters = []
        for filter_index in range(nr_filter_dict[curr_layer_name]):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
            logging.info('Processing filter %d' % filter_index)
            start_time = time.time()

            layer_output = layer_dict[curr_layer_name].output

            if k_backend.image_data_format() == 'channels_first':
                loss = k_backend.mean(layer_output[:, filter_index, :, :])
            else:
                loss = k_backend.mean(layer_output[:, :, :, filter_index])

            # compute the gradient of the input picture wrt this loss
            grads = k_backend.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = k_backend.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 10.

            # we start from a gray image with some noise
            if use_real_img:
                img = load_image_absolute(path_image)
                img = preprocess(img, model_name)
                input_img_data = img[None, :, :, :]
            else:
                # we start from a gray image with some random noise
                if k_backend.image_data_format() == 'channels_first':
                    input_img_data = np.random.random((1, 3, img_height, img_width))
                else:
                    input_img_data = np.random.random((1, img_height, img_width, 3))
                    input_img_data = (input_img_data - 0.5) * 20 + 128

            input_img_data = np.asarray(input_img_data, dtype=np.float32)

            # run gradient ascent for 20 steps
            loss_value = 0
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # logging.info('Current loss value:', loss_value)
                logging.info('Current loss value: {:.4f} '.format(loss_value))

            img = deprocess_image(input_img_data[0])

            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kept_filters.append((img, loss_value))

            end_time = time.time()
            logging.info('Filter %d processed in %ds' % (filter_index, end_time - start_time))

        # we will stich the best 16 filters on a 4 x 4 grid.
        n = 4

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin

        stitched_filters = np.zeros((height, width, 3))
        # stitched_filters = np.zeros((height, width))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                stitched_filters[(img_height + margin) * j: (img_height + margin) * j + img_height,
                                 (img_width + margin) * i: (img_width + margin) * i + img_width, :] = img

        # save the result to disk
        imsave('./visualize_cnn/' + curr_layer_name + '_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

    return 0


if __name__ == '__main__':
    # Logging
    logging.basicConfig(level=logging.INFO)
    # Parser
    parser = argparse.ArgumentParser(description='CONV Filter Visualisation')
    parser.add_argument('-p', help='path to model.h5 file', dest='model_path', type=str, default=None)
    parser.add_argument('-i', help='path to imagefile', dest='image_path', type=str, default=None)
    parser.add_argument('-r', help='use real image', dest='use_real_img', type=s2b, default='false')
    args = parser.parse_args()

    # C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_electron_nadam_val_split/model-099.h5
    # C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_nvidia_val_berlin/model-053.h5

    # C:/Users/timmy/Documents/Dev/Thesis/code/rec_data/jungle/IMG/center_2018_01_19_12_10_06_223.jpg

    if args.model_path:
        if args.use_real_img:
            if args.image_path:
                main(args)
            else:
                logging.info('No image path found. Exit')
        else:
            main(args)
    else:
        logging.info('No model specified. Exit.')
