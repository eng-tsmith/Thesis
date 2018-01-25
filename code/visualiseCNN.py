from keras.models import load_model
from preprocessing import load_image, preprocess
import numpy as np
import time
from keras import backend as K
import cv2
from scipy.misc import imsave
# Manual deactivation of learning mode for backend functions
K.set_learning_phase(0)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def main():
    # https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    # 'conv_layer_1': --> max Filter: 16
    # 'conv_layer_2': --> max Filter: 16
    # 'conv_layer_3': --> max Filter: 32
    # 'conv_layer_4': --> max Filter: 32
    # 'conv_layer_5': --> max Filter: 64
    # 'conv_layer_6': --> max Filter: 64
    # 'conv_layer_7': --> max Filter: 128
    # 'conv_layer_8': --> max Filter: 128

    # # coole: l:2 f:5
    # layer_name = 'conv_layer_3'
    # filter_index = 5


    path_weights = 'C:/ProgramData/Thesis/code/logs/1516790366.9310215/model-038.h5'
    path_image = '.\\data_0\\IMG\\center_2017_07_21_14_27_07_549.jpg'

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
    data_dir = '.\\rec_data'

    # dimensions of the generated pictures for each filter.
    img_height = 66
    img_width = 200

    # build model
    model = load_model(path_weights)
    model.summary()

    cv2.namedWindow('CNN layer visualisation', cv2.WINDOW_NORMAL)

    for curr_layer_name in layers:
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # this is the placeholder for the input images
        input_img = model.input

        img = load_image(data_dir, path_image)
        img = preprocess(img)
        cv2.imshow('CNN layer visualisation', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(2000)

        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        kept_filters = []
        for filter_index in range(nr_filter_dict[curr_layer_name]):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
            print('Processing filter %d' % filter_index)
            start_time = time.time()

            layer_output = layer_dict[curr_layer_name].output
            loss = K.mean(layer_output[:, :, :, filter_index])

            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some noise
            # input_img_data = np.random.random((1, img_height, img_width, 3)) * 20 + 128.
            img = load_image(data_dir, path_image)
            img = preprocess(img)
            input_img_data = img[None, :, :, :]

            # run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                # # if loss_value <= 0.:
                #     # some filters get stuck to 0, we can skip them
                #     break
                #
                # # decode the resulting input image
                # if loss_value > 0:
                #     img = deprocess_image(input_img_data[0])
                #     kept_filters.append((img, loss_value))

            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

            end_time = time.time()
            print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

        # we will stich the best 64 filters on a 8 x 8 grid.
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

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                stitched_filters[(img_height + margin) * j: (img_height + margin) * j + img_height,
                (img_width + margin) * i: (img_width + margin) * i + img_width, :] = img

        # cv2.imshow('CNN layer visualisation', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(10000)
        # save the result to disk
        imsave('./visualize_cnn/' + curr_layer_name + '_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

    return 0


if __name__ == '__main__':
    main()
