from keras.models import load_model
from preprocessing import load_image, preprocess
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import backend as K
import cv2
from scipy.misc import imsave
# Manual deactivation of learning mode for backend functions
K.set_learning_phase(0)


def display_activations(activation_maps):
    """
    Source: https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
    :param activation_maps:

    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
    plt.show()


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    """
    Source: https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
    :param model:
    :param model_inputs:
    :param print_shape_only:
    :param layer_name:
    """
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    print(outputs)
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]

    print(funcs)

    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


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


def visual_activation():
    data_dir = '.\\rec_data'
    path_weights = 'C:\\Users\\timmy\\Documents\\Dev\\Thesis\\code\\logs\\FINAL_electron_nadam_val_split/model-099.h5'
    path_image = '.\\berlin\\IMG\\center_2018_01_19_13_23_52_018.jpg'

    # Load image
    img = load_image(data_dir, path_image)
    img = preprocess(img)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    cv2.namedWindow('CNN layer visualisation', cv2.WINDOW_NORMAL)
    cv2.imshow('CNN layer visualisation', img/255)
    cv2.waitKey(2000)

    img = np.asarray(img, dtype=np.float32)
    img = img[None, :, :, :]



    # build model
    model = load_model(path_weights)
    model.summary()

    # Get Activation
    a = get_activations(model, img, print_shape_only=True)  # with just one sample.
    display_activations(a)


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

    use_real_img = True

    path_weights = 'C:\\Users\\timmy\\Documents\\Dev\\Thesis\\code\\logs\\FINAL_electron_nadam_val_split/model-099.h5'
    path_image = '.\\berlin\\IMG\\center_2018_01_19_13_23_52_018.jpg'

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
    img_height = 64
    img_width = 64

    # build model
    model = load_model(path_weights)
    model.summary()

    cv2.namedWindow('CNN layer visualisation', cv2.WINDOW_NORMAL)

    for curr_layer_name in layers:
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # this is the placeholder for the input images
        input_img = model.input

        if use_real_img:
            img = load_image(data_dir, path_image)
            img = preprocess(img)
            img_show = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            cv2.imshow('CNN layer visualisation', img_show/255)
            cv2.waitKey(2000)
        else:
            img = np.random.random((img_height, img_width, 3)) * 20 + 128.

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
            if use_real_img:
                img = load_image(data_dir, path_image)
                img = preprocess(img)
            else:
                img = np.random.random((img_height, img_width, 3)) * 20 + 128.

            input_img_data = img[None, :, :, :]

            # run gradient ascent for 20 steps
            for i in range(1000):
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
    # visual_activation()
