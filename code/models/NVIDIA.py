from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Model


def build_model(args, input_shape):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """

    inputs = Input(shape=input_shape, name='main_input')
    norm = Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape)(inputs)

    conv1 = Conv2D(24, (5, 5), strides=(2, 2), activation='elu')(norm)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), activation='elu')(conv1)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation='elu')(conv2)
    conv4 = Conv2D(64, (3, 3), activation='elu')(conv3)
    conv5 = Conv2D(64, (3, 3), activation='elu')(conv4)

    dropout = Dropout(args.keep_prob)(conv5)

    flat = Flatten()(dropout)

    dense1 = Dense(100, activation='elu')(flat)
    dense2 = Dense(50, activation='elu')(dense1)
    dense3 = Dense(10, activation='elu')(dense2)
    output = Dense(1)(dense3)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model
