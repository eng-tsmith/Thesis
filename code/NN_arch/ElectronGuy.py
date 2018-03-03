from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential
from preprocessing import normalize_img
import logging


def build_model(args, input_shape):
    """
    https://github.com/electroncastle/behavioral_cloning/blob/master/model.py
    """
    name = 'ElectronGuy'
    l2_weight_decay = args.l2_weight_decay
    dropout = args.drop_prob

    inputs = Input(shape=input_shape, name='main_input')

    norm = Lambda(normalize_img)(inputs)

    conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_1')(norm)
    max1 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_1')(conv1)
    conv2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_2')(max1)
    max2 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_2')(conv2)
    conv3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_3')(max2)
    conv4 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_4')(conv3)
    max4 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_4')(conv4)

    dropout1 = Dropout(dropout, name='dropout_layer_1')(max4)

    conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_5')(dropout1)
    conv6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_6')(conv5)
    max6 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_6')(conv6)

    dropout2 = Dropout(dropout, name='dropout_layer_2')(max6)

    conv7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_7')(dropout2)
    conv8 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='elu', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_8')(conv7)
    max8 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_8')(conv8)

    dropout3 = Dropout(dropout, name='dropout_layer_3')(max8)

    flat = Flatten(name='flatten_layer')(dropout3)
    dense1 = Dense(128, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_1')(flat)

    dropout4 = Dropout(dropout, name='dropout_layer_4')(dense1)

    dense2 = Dense(96, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_2')(dropout4)

    dropout5 = Dropout(dropout, name='dropout_layer_5')(dense2)

    dense3 = Dense(64, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_3')(dropout5)
    dense4 = Dense(10, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_4')(dense3)

    if args.label_dim == 1:
        output = Dense(1, activation='linear', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_5')(dense4)
    elif args.label_dim == 2:
        output = Dense(2, activation='linear', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_5')(dense4)
    else:
        logging.info("unknown output dimension. cannot build!")
        return -1, -1

    # inputs = Input(shape=input_shape, name='main_input')
    #
    # norm = Lambda(normalize_img)(inputs)
    #
    # conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_layer_1')(norm)
    # max1 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_1')(conv1)
    # conv2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_layer_2')(max1)
    # max2 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_2')(conv2)
    # conv3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_layer_3')(max2)
    # conv4 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_layer_4')(conv3)
    # max4 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_4')(conv4)
    #
    # dropout1 = Dropout(dropout, name='dropout_layer_1')(max4)
    #
    # conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu', name='conv_layer_5')(dropout1)
    # conv6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu', name='conv_layer_6')(conv5)
    # max6 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_6')(conv6)
    #
    # dropout2 = Dropout(dropout, name='dropout_layer_2')(max6)
    #
    # conv7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu', name='conv_layer_7')(dropout2)
    # conv8 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='elu', name='conv_layer_8')(conv7)
    # max8 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_8')(conv8)
    #
    # dropout3 = Dropout(dropout, name='dropout_layer_3')(max8)
    #
    # flat = Flatten(name='flatten_layer')(dropout3)
    # dense1 = Dense(128, activation='elu', name='fc_layer_1')(flat)
    #
    # dropout4 = Dropout(dropout, name='dropout_layer_4')(dense1)
    #
    # dense2 = Dense(96, activation='elu', name='fc_layer_2')(dropout4)
    #
    # dropout5 = Dropout(dropout, name='dropout_layer_5')(dense2)
    #
    # dense3 = Dense(64, activation='elu', name='fc_layer_3')(dropout5)
    # dense4 = Dense(10, activation='elu', name='fc_layer_4')(dense3)
    #
    # if args.label_dim == 1:
    #     output = Dense(1, activation='linear', name='fc_layer_5')(dense4)
    # elif args.label_dim == 2:
    #     output = Dense(2, activation='linear', name='fc_layer_5')(dense4)
    # else:
    #     logging.info("unknown output dimension. cannot build!")
    #     return -1, -1

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model, name
