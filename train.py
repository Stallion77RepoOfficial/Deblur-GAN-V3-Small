# train.py

import tensorflow as tf
from model import build_small_model
from dataset import load_dataset

def train(epochs, batch_size, dataset_path):
    # Modeli oluştur
    model = build_small_model(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='mae')

    # Veri setini yükle
    dataset = load_dataset(dataset_path, batch_size, img_size=(256, 256))

    # Eğit
    model.fit(dataset, epochs=epochs)

    # Modeli kaydet
    model.save('/Users/berkegulacar/Downloads/deblurganv2/small_model.h5')

if __name__ == "__main__":
    train(epochs=200, batch_size=32, dataset_path='/Users/berkegulacar/Downloads/deblurganv2/GOPRO_Large')