# models/unet.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def build_optimized_unet(input_shape=(256, 256, 3), num_filters=32):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(num_filters, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(num_filters*2, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(num_filters*2, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(num_filters*4, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(num_filters*4, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(num_filters*8, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(num_filters*8, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Conv2D(num_filters*4, (2, 2), padding='same')(u5)
    merge5 = Concatenate()([u5, c3])
    c5 = Conv2D(num_filters*4, (3, 3), padding='same')(merge5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(num_filters*4, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(num_filters*2, (2, 2), padding='same')(u6)
    merge6 = Concatenate()([u6, c2])
    c6 = Conv2D(num_filters*2, (3, 3), padding='same')(merge6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(num_filters*2, (3, 3), padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(num_filters, (2, 2), padding='same')(u7)
    merge7 = Concatenate()([u7, c1])
    c7 = Conv2D(num_filters, (3, 3), padding='same')(merge7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(num_filters, (3, 3), padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    outputs = Conv2D(3, (1, 1), activation='tanh')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
