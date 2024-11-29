# dataset/dataset.py

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(config):
    dataset_path = config['data']['dataset_path']
    batch_size = config['training']['batch_size']
    img_height, img_width = config['data']['img_size']
    buffer_size = config['data']['buffer_size']
    num_parallel_calls = config['data']['num_parallel_calls']

    def preprocess_image(file_path):
        # Görüntüyü yükle ve yeniden boyutlandır
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_height, img_width])
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0  # Normalize to [-1, 1]

        # Veri artırma
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Bulanıklaştırma
        blurred_image = apply_blur(image)

        return blurred_image, image

    def apply_blur(image):
        # Basit bir Gauss bulanıklaştırma işlemi
        image = tf.expand_dims(image, axis=0)  # [H, W, C] -> [1, H, W, C]
        kernel = tf.constant([[1/16, 2/16, 1/16],
                              [2/16, 4/16, 2/16],
                              [1/16, 2/16, 1/16]],
                             dtype=tf.float32, shape=[3, 3, 1, 1])
        kernel = tf.repeat(kernel, repeats=3, axis=2)  # RGB kanalları için
        blurred = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'SAME')
        blurred = tf.squeeze(blurred, axis=0)  # [1, H, W, C] -> [H, W, C]
        return blurred

    # Görüntü dosyalarını toplama
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, file)
                image_files.append(img_path)

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Eğitim ve Doğrulama setlerine ayırma
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    val_batches = int(config['training']['validation_split'] * total_batches)
    train_dataset = dataset.skip(val_batches)
    val_dataset = dataset.take(val_batches)

    return train_dataset, val_dataset
