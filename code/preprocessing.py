import cv2
import os
import numpy as np
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
import sklearn.utils as skutil
import logging
from random import randint
from matplotlib2tikz import save as tikz_save


# NVIDIA
IMAGE_HEIGHT_NVIDIA, IMAGE_WIDTH_NVIDIA, IMAGE_CHANNELS_NVIDIA = 66, 200, 3
INPUT_SHAPE_NVIDIA = (IMAGE_HEIGHT_NVIDIA, IMAGE_WIDTH_NVIDIA, IMAGE_CHANNELS_NVIDIA)

# ELECTRON
IMAGE_HEIGHT_ELECTRON, IMAGE_WIDTH_ELECTRON, IMAGE_CHANNELS_ELECTRON = 64, 64, 3
INPUT_SHAPE_ELECTRON = (IMAGE_HEIGHT_ELECTRON, IMAGE_WIDTH_ELECTRON, IMAGE_CHANNELS_ELECTRON)


def plot_image(image_display):
    """
    Plot the image. Used for testing functions.
    :param image_display:
    """
    plt.imshow(np.asarray(image_display, dtype='uint8'))
    plt.show()


def process_img_for_visualization(image, angle=None, pred_angle=None):
    """
    Used by visualize_dataset method to format image prior to displaying. Converts colorspace back to original BGR,
    applies text to display steering angle and frame number (within batch to be visualized), and applies lines
    representing steering angle and model-predicted steering angle (if available) to image.
    Source:
    https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py
    """
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    h, w = img.shape[0:2]

    # apply a line representing the steering angle
    if angle is not None:
        cv2.line(img, (int(w/2), int(h)), (int(w/2+angle*w/4), int(h/2)), (0, 255, 0), thickness=4)

    if pred_angle is not None:
        cv2.line(img, (int(w/2), int(h)), (int(w/2+pred_angle*w/4), int(h/2)), (0, 0, 255), thickness=4)
    return img


def center_val_data(x_in):
    """
    Only consider center images
    :param x_in:
    :return:
    """
    x_out = x_in[:, 0]

    return x_out


def l_c_r_data(x_in, y_in, angle_adj=0.25):
    """
    Reshape array and adjust angle of left and right images
    :param x_in: center left right
    :param y_in:
    :param angle_adj:
    :return:
    """
    if y_in.ndim == 1:
        x_out = np.reshape(x_in, x_in.shape[0]*x_in.shape[1])
        y_out = np.append(y_in, [y_in + angle_adj, y_in - angle_adj])
    elif y_in.ndim == 2:
        x_out = np.reshape(x_in, x_in.shape[0]*x_in.shape[1])
        y_1 = np.append(y_in[:, 0], [y_in[:, 0] + angle_adj, y_in[:, 0] - angle_adj])
        y_2 = np.append(y_in[:, 1], [y_in[:, 1], y_in[:, 1]])
        y_out = np.column_stack((y_1, y_2))
    else:
        logging.info("ERROR! Unknown Dimension. Look at y dim")
        return 0, 0
    return x_out, y_out


def flatten_data(x_in, y_in, num_bins=25, print_enabled=False, plot_enabled=False):
    """
    print a histogram to see which steering angle ranges are most overrepresented
    print histogram again to show more even distribution of steering angles

    Source : https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py
    :param x_in:
    :param y_in:
    :param num_bins:
    :param print_enabled:
    :param plot_enabled:
    :return:
    """
    if y_in.ndim == 1:
        y_curr = y_in
    elif y_in.ndim == 2:
        y_curr = y_in[:, 0]
        y_speed = y_in[:, 1]

        avg_samples_per_bin = y_speed.size / num_bins
        hist, bins = np.histogram(y_speed, num_bins)
        if print_enabled:
            width = 0.7 * (bins[1] - bins[0])
            cent = (bins[:-1] + bins[1:]) / 2
            plt.bar(cent, hist, align='center', width=width)
            plt.plot((np.min(y_speed), np.max(y_speed)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
            if plot_enabled:
                plt.xlabel('Geschwindigkeiten')
                plt.ylabel('Anzahl Bilder')
                tikz_save('label_speed_full.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
                plt.show()
            else:
                plt.show()
    else:
        logging.info("ERROR! Unknown Dimension. Look at y dim")
        return 0, 0

    avg_samples_per_bin = y_curr.size / num_bins
    hist, bins = np.histogram(y_curr, num_bins)
    if print_enabled:
        width = 0.7 * (bins[1] - bins[0])
        cent = (bins[:-1] + bins[1:]) / 2
        plt.bar(cent, hist, align='center', width=width)
        plt.plot((np.min(y_curr), np.max(y_curr)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
        if plot_enabled:
            plt.xlabel('Lenkwinkel')
            plt.ylabel('Anzahl Bilder')
            tikz_save('label_lenk_full.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.show()
        else:
            plt.show()

    # determine keep probability for each bin: if below avg_samples_per_bin, keep all;
    # otherwise keep prob is proportional to number of samples above the average,
    # so as to bring the number of samples for that bin down to the average
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1. / (hist[i] / target))
    remove_list = []
    for i in range(len(y_curr)):
        for j in range(num_bins):
            if bins[j] < y_curr[i] <= bins[j + 1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)

    y_out = np.delete(y_in, remove_list, 0)
    x_out = np.delete(x_in, remove_list, 0)

    # print histogram again to show more even distribution of steering angles
    if print_enabled:
        if y_in.ndim == 1:
            y_curr = y_out
        elif y_in.ndim == 2:
            y_curr = y_out[:, 0]
            y_speed = y_out[:, 1]

            hist, bins = np.histogram(y_speed, num_bins)
            if print_enabled:
                width = 0.7 * (bins[1] - bins[0])
                cent = (bins[:-1] + bins[1:]) / 2
                plt.bar(cent, hist, align='center', width=width)
                plt.plot((np.min(y_speed), np.max(y_speed)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
                if plot_enabled:
                    plt.xlabel('Geschwindigkeiten')
                    plt.ylabel('Anzahl Bilder')
                    tikz_save('label_speed_flat.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
                    plt.show()
                else:
                    plt.show()

        hist, bins = np.histogram(y_curr, num_bins)
        width = 0.7 * (bins[1] - bins[0])
        cent = (bins[:-1] + bins[1:]) / 2
        plt.bar(cent, hist, align='center', width=width)
        plt.plot((np.min(y_curr), np.max(y_curr)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

        if plot_enabled:
            plt.xlabel('Lenkwinkel')
            plt.ylabel('Anzahl Bilder')
            tikz_save('label_lenk_flat.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.show()
        else:
            plt.show()

    logging.info('Data set distribution flattened. Data set size reduced from {} to {}'.format(x_in.size, x_out.size))
    return x_out, y_out


def load_image_absolute(image_file):
    """
    Load RGB images from a file. Uses absolute path.
    :param image_file:
    :return:
    """
    # Transform to RGB, so that preprocessing is same for train and realtime (unity sends rgb image)
    return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file. Takes data_dir as root folder so that the paths in csv file are considered as relative
    :param data_dir:
    :param image_file:
    :return:
    """
    path = os.path.join(data_dir,
                        image_file.strip().split("\\")[-3],
                        image_file.strip().split("\\")[-2],
                        image_file.strip().split("\\")[-1])
    # Transform to RGB, so that preprocessing is same for train and realtime (unity sends rgb image)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    :param image:
    :return:
    """
    crop_top = 60
    crop_bot = 23
    h, w = image.shape[0], image.shape[1]

    return image[crop_top:h - crop_bot, 0:w]


def resize(image, model_name):
    """
    Resize the image to the input shape used by the network model
    :param image: image array
    :param model_name: name of model for proper dim
    :return:
    """
    if model_name == 'nvidia':
        img = cv2.resize(image, (IMAGE_WIDTH_NVIDIA, IMAGE_HEIGHT_NVIDIA), cv2.INTER_AREA)
    elif model_name == 'electron':
        img = cv2.resize(image, (IMAGE_WIDTH_ELECTRON, IMAGE_HEIGHT_ELECTRON), cv2.INTER_AREA)
    else:
        logging.info('Model not existent')
        return -1
    return img


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA Paper does).
    :param image:
    :return:
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def normalize_img(image):
    img = image/255.0 - 0.5
    return img


def denormalize_img(image):
    img = (image + 0.5) * 255.0
    return img


def preprocess(image, model_name):
    """
    Combine all preprocess functions into one
    """
    # Crop
    image = crop(image)

    # Blur
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    # Resize
    image = resize(image, model_name)
    image = np.asarray(image, dtype=np.float32)

    # Normalize moved to NN
    # image = normalize_img(image)  # Move to NN for GPU computation

    # NVIDIA Paper
    image = rgb2yuv(image)
    return image


def random_flip(image, steering_angle):
    """
    Randomly mirrors the image horizontally, and accordingly the steering angle
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shifts the image vertically and horizontally (translation)
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    if np.random.rand() < 0.5:
        if np.random.rand() < 0.5:
            image.setflags(write=1)
            h, w = image.shape[:2]
            [x1, x2] = np.random.choice(w, 2, replace=False)
            if (x2 - x1) != 0:
                m = h / (x2 - x1)
                t = - m * x1
                for i in range(h):
                    c = int((i - t) / m)
                    image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
        else:
            image.setflags(write=1)
            h, w = image.shape[:2]
            [x1, x2] = np.random.choice(h, 2, replace=False)
            if (x2 - x1) != 0:
                m = w / (x2 - x1)
                t = - m * x1
                for i in range(w):
                    c = int((i - t) / m)
                    image[:c, i, :] = (image[:c, i, :] * .5).astype(np.int32)
    return image


def random_brightness(image):
    """
    Randomly adjusts brightness of the image
    """
    if np.random.rand() < 0.5:
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        value = randint(-25, 25)

        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = 0 - value
            v[v < lim] = 0
            v[v >= lim] = v[v >= lim] + value

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    return image


def normalize_speed(speed):
    """
    Normalize speed from m/s to [-1, 1]
    :param speed:
    :return:
    """
    max_speed = 60.0  # 216 km/h = 60 m/s
    return 2 * speed / max_speed - 1.0


def denormalize_speed(speed):
    """
    Denormalize speed from [-1, 1] to m/s
    :param speed:
    :return:
    """
    max_speed = 60.0  # 216 km/h = 60 m/s
    return (speed + 1.0) * max_speed / 2


def augment(data_dir, image_path, steering_angle):
    """
    Generate an augmented image and adjusts steering angle.
    """
    image = load_image(data_dir, image_path)

    image, steering_angle = random_flip(image, steering_angle)
    # image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)

    return image, steering_angle


def batch_generator(data_dir, x_in, y_in, batch_size, label_dim, is_training, model_name):
    """
    Generator for training/ valdiation data
    :param data_dir:
    :param x_in:
    :param y_in:
    :param batch_size:
    :param label_dim:
    :param is_training:
    :param model_name:
    """
    curr_image = 0
    n_images = x_in.size

    while True:
        if curr_image > (n_images - batch_size):
            curr_image = 0
        if curr_image == 0:
            x_in, y_in = skutil.shuffle(x_in, y_in)

        future_index = curr_image + batch_size

        x_data = x_in[curr_image:future_index]
        y_data = y_in[curr_image:future_index]

        if model_name == 'nvidia':
            x_batch = np.empty([batch_size, IMAGE_HEIGHT_NVIDIA, IMAGE_WIDTH_NVIDIA, IMAGE_CHANNELS_NVIDIA])
        elif model_name == 'electron':
            x_batch = np.empty([batch_size, IMAGE_HEIGHT_ELECTRON, IMAGE_WIDTH_ELECTRON, IMAGE_CHANNELS_ELECTRON])
        else:
            logging.info('Model does not exist')
            return -1

        if label_dim == 1:
            y_batch = np.empty(batch_size)
        elif label_dim == 2:
            y_batch = np.empty([batch_size, 2])
        else:
            logging.info("ERROR! Unknown Dimension. Look at label dim")
            return -1

        for sample_index in range(x_data.size):
            # Gather batch for steering angle only
            if label_dim == 1:
                # Only do data augmentation for training data with a probability of 60%
                if is_training and np.random.rand() < 0.6:
                    image, label_steer = augment(data_dir, x_data[sample_index], y_data[sample_index][0])
                else:
                    image = load_image(data_dir, x_data[sample_index])
                    label_steer = y_data[sample_index][0]

                # Preprocessing goes for all data
                x_batch[sample_index] = preprocess(image, model_name)
                y_batch[sample_index] = np.asarray([label_steer])

            # Gather batch for steering angle and speed
            elif label_dim == 2:
                # Only do data augmentation for training data with a probability of 60%
                if is_training and np.random.rand() < 0.6:
                    image, label_steer = augment(data_dir, x_data[sample_index], y_data[sample_index][0])
                    label_speed = normalize_speed(y_data[sample_index][1])
                else:
                    image = load_image(data_dir, x_data[sample_index])
                    label_steer = y_data[sample_index][0]
                    label_speed = normalize_speed(y_data[sample_index][1])

                # Preprocessing goes for all data
                x_batch[sample_index] = preprocess(image, model_name)
                y_batch[sample_index] = np.asarray([label_steer, label_speed])

        # Keeps track of which image has already been seen
        curr_image += batch_size

        # Yield as numpy array for tensorflow backend
        x_batch = np.asarray(x_batch, dtype='float32')
        y_batch = np.asarray(y_batch, dtype='float32')

        if label_dim == 1:
            yield (x_batch, y_batch)
        elif label_dim == 2:
            y_batch1 = y_batch[:, 0]
            y_batch2 = y_batch[:, 1]
            yield (x_batch, [y_batch1, y_batch2])
        else:
            logging.info('Wrong dim')
            return -1
