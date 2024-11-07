# model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Add
from tensorflow.keras.models import Model

def build_small_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    # Basit artık blok
    def res_block(x, filters):
        res = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
        res = LeakyReLU(0.2)(res)
        res = Conv2D(filters, kernel_size=3, strides=1, padding='same')(res)
        res = Add()([res, x])
        return res

    x = res_block(x, 16)
    x = res_block(x, 16)

    outputs = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model