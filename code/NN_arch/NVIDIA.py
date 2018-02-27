from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Model
from preprocessing import normalize_img
from keras.regularizers import l2
import logging


def build_model(args, input_shape):
    """
    NVIDIA Paper:
    End to End Learning for Self-Driving Cars
    https://arxiv.org/abs/1604.07316
    """
    name = 'NVIDIA'

    inputs = Input(shape=input_shape, name='main_input')

    norm = Lambda(normalize_img)(inputs)

    conv1 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros', name='conv1')(norm)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros', name='conv2')(conv1)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros', name='conv3')(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros', name='conv4')(conv3)
    conv5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros', name='conv5')(conv4)

    # dropout = Dropout(args.drop_prob, name='dropout')(conv5)

    flat = Flatten(name='flatten_layer')(conv5)

    dense1 = Dense(100, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros',  name='fc1')(flat)
    dense2 = Dense(50,  activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros',  name='fc2')(dense1)
    dense3 = Dense(10,  activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='RandomUniform', bias_initializer='zeros',  name='fc3')(dense2)

    if args.label_dim == 1:
        output = Dense(1, name='output')(dense3)
    elif args.label_dim == 2:
        output = Dense(2, name='output')(dense3)
    else:
        logging.info("unknown output dimension. cannot build!")
        return -1, -1

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model, name
