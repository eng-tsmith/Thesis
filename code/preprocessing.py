import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import logging


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 66, 3 #66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def plot_image(image_display):
    """
    Plot the image. Used for testing functions.
    :param image_display:
    """
    plt.imshow(np.asarray(image_display, dtype='uint8'))
    plt.show()


def center_val_data(X_in):
    """
    Only consider center images
    :param X_in:
    :return:
    """
    X_out = X_in[:, 0]

    return X_out


def l_c_r_data(x_in, y_in, angle_adj=0.2):
    """
    Reshape array and adjust angle of left and right images
    :param x_in:
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


def flatten_data(X_in, y_in, num_bins=25, print_enabled=False):
    """
    print a histogram to see which steering angle ranges are most overrepresented
    print histogram again to show more even distribution of steering angles

    Source : https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py
    :param X_in:
    :param y_in:
    :param num_bins:
    :param print_enabled:
    :return:
    """
    if y_in.ndim == 1:
        y_curr = y_in
    elif y_in.ndim == 2:
        y_curr = y_in[:, 0]
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
        plt.show()

    # determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
    # to number of samples above the average, so as to bring the number of samples for that bin down to the average
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
            if y_curr[i] > bins[j] and y_curr[i] <= bins[j + 1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)

    y_out = np.delete(y_in, remove_list, 0)
    X_out = np.delete(X_in, remove_list, 0)

    # print histogram again to show more even distribution of steering angles
    if print_enabled:
        if y_in.ndim == 1:
            y_curr = y_out
        elif y_in.ndim == 2:
            y_curr = y_out[:, 0]

        hist, bins = np.histogram(y_curr, num_bins)
        plt.bar(cent, hist, align='center', width=width)
        plt.plot((np.min(y_curr), np.max(y_curr)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
        plt.show()

    logging.info('Data set distribution flattened. Data set size reduced from {} to {}'.format(X_in.size, X_out.size))
    return X_out, y_out


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    :param data_dir:
    :param image_file:
    :return:
    """
    #return mpimg.imread(os.path.join(data_dir, image_file.strip()))
    return mpimg.imread(os.path.join(data_dir, image_file.strip().split("\\")[-3], image_file.strip().split("\\")[-2], image_file.strip().split("\\")[-1]))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    :param image:
    :return:
    """
    h, w = image.shape[0], image.shape[1]
    return image[60:h - 23, 0:w]#  image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    :param image:
    :return:
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA Paper does).
    :param image:
    :return:
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    # Normalize
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0 - 0.5
    #logging.info(image)
    # This may become interesting when taking images from real camera
    #image = rgb2yuv(image
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Take a random image (center, left right) and adjust steering angle
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly mirrors the image left <-> right, and accordingly the steering angle
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


def random_shadow_old(image):
    """
    Generates and adds random shadow. Not used anymore
    """
    height, width = image.shape[:2]
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = int(width * np.random.rand()), 0
    x2, y2 = int(width * np.random.rand()), height
    xm, ym = np.mgrid[0:height, 0:width]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    image.setflags(write=1)
    h, w = image.shape[:2]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image


def random_brightness(image):
    """
    Randomly adjusts brightness of the image
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def normalize_speed(speed):
    """
    Normalize speed from MPH to [-0.5, 0.5]
    :param speed:
    :return:
    """
    return speed / 31.0 - 0.5 #TODO check if mph or m/s


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    #image = load_image(data_dir, image_path)
    image, steering_angle = random_flip(image, steering_angle)
    #image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)

    return image, steering_angle


def augment2(data_dir, image_path, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image = load_image(data_dir, image_path)
    image, steering_angle = random_flip(image, steering_angle)
    #image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)

    return image, steering_angle


def batch_generator_old(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generator for training/ valdiation data. OLD, not used anymore
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # augmentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = np.array([steering_angle])
            i += 1
            if i == batch_size:
                break
        yield images, steers


def batch_generator2(data_dir, x_in, y_in, batch_size, is_training):
    """
    Generator for training/ valdiation data
    :param data_dir:
    :param x_in:
    :param y_in:
    :param batch_size:
    :param is_training:
    """
    curr_image = 0
    n_images = x_in.size

    while True:
        if curr_image > n_images:
            curr_image = 0
        if curr_image == 0:
            x_in, y_in = shuffle(x_in, y_in)

        future_index = curr_image + batch_size

        x_data = x_in[curr_image:future_index]
        y_data = y_in[curr_image:future_index]

        x_batch = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        if y_in.ndim == 1:
            y_batch = np.empty(batch_size)
        elif y_in.ndim == 2:
            y_batch = np.empty([batch_size, 2])
        else:
            logging.info("ERROR! Unknown Dimension. Look at y dim")
            return 0, 0

        for sample_index in range(x_data.size):
            # Only do data augmentation for training data with a probability of 60%
            if y_in.ndim == 1:
                if is_training and np.random.rand() < 0.6:
                    image, label_steer = augment2(data_dir, x_data[sample_index], y_data[sample_index])
                else:
                    image = load_image(data_dir, x_data[sample_index])
                    label_steer = y_data[sample_index]

                # Preprocessing goes for all data
                x_batch[sample_index] = preprocess(image)
                y_batch[sample_index] = np.array([label_steer])

            elif y_in.ndim == 2:
                if is_training and np.random.rand() < 0.6:
                    image, label_steer = augment2(data_dir, x_data[sample_index], y_data[sample_index][0])
                else:
                    image = load_image(data_dir, x_data[sample_index])
                    label_steer = y_data[sample_index][0]

                # Preprocessing goes for all data
                x_batch[sample_index] = preprocess(image)
                label_speed = normalize_speed(y_data[sample_index][1])
                y_batch[sample_index] = np.array([label_steer, label_speed])

        # Keeps track of which image has already been seen
        curr_image += batch_size

        # Yield as numpy array for tensorflow backend
        x_batch = np.asarray(x_batch, dtype='float32')
        y_batch = np.asarray(y_batch, dtype='float32')

        yield (x_batch, y_batch)
