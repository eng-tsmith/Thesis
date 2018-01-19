from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential


def build_model(args, input_shape):
    """
    https://github.com/electroncastle/behavioral_cloning/blob/master/model.py
    """
    name = 'ElectronGuy'
    l2_weight_decay = 1e-5

    inputs = Input(shape=input_shape, name='main_input')
    #norm = Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape, name='norm_layer')(inputs)
    #norm = BatchNormalization(input_shape=input_shape, axis=1, name='norm_layer')(inputs)

    conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_1')(inputs)
    act1 = Activation('relu', name='act_layer_1')(conv1)
    max1 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_1')(act1)
    conv2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_2')(max1)
    act2 = Activation('relu', name='act_layer_2')(conv2)
    max2 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_2')(act2)

    conv3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_3')(max2)
    act3 = Activation('relu', name='act_layer_3')(conv3)
    conv4 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_4')(act3)
    act4 = Activation('relu', name='act_layer_4')(conv4)
    max4 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_4')(act4)
    dropout1 = Dropout(0.5, name='dropout_layer_1')(max4)

    conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_5')(dropout1)
    act5 = Activation('elu', name='act_layer_5')(conv5)
    conv6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_6')(act5)
    act6 = Activation('elu', name='act_layer_6')(conv6)
    max6 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_6')(act6)
    dropout2 = Dropout(0.5, name='dropout_layer_2')(max6)

    conv7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_7')(dropout2)
    act7 = Activation('elu', name='act_layer_7')(conv7)
    conv8 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_weight_decay), name='conv_layer_8')(act7)
    act8 = Activation('elu', name='act_layer_8')(conv8)
    max8 = MaxPooling2D(pool_size=(2, 2), name='pool_layer_8')(act8)
    dropout3 = Dropout(0.5, name='dropout_layer_3')(max8)

    flat = Flatten(name='flatten_layer')(dropout3)
    dense1 = Dense(128, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_1')(flat)
    dropout4 = Dropout(0.5, name='dropout_layer_4')(dense1)
    dense2 = Dense(96, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_2')(dropout4)
    dropout5 = Dropout(0.5, name='dropout_layer_5')(dense2)
    dense3 = Dense(64, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_3')(dropout5)
    dense4 = Dense(10, activation='elu', kernel_regularizer=l2(l2_weight_decay), name='fc_layer_4')(dense3)
    output = Dense(2, activation='linear', kernel_regularizer=l2(0.0), name='fc_layer_5')(dense4)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    # model = Sequential()
    #
    # model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(.5))
    #
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('elu'))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(.5))
    #
    # model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('elu'))
    # model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Activation('elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(.5))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='elu', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Dropout(.5))
    # model.add(Dense(96, activation='elu', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Dropout(.5))
    # model.add(Dense(64, activation='elu', kernel_regularizer=l2(l2_weight_decay)))
    # model.add(Dense(10, activation='elu', kernel_regularizer=l2(l2_weight_decay)))
    #
    # model.add(Dense(2, activation='linear', kernel_regularizer=l2(0.0)))
    # model.summary()

    return model, name
