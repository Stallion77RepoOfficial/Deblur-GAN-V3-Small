# dataset.py

import tensorflow as tf
import os

def load_dataset(image_path, batch_size, img_size=(256, 256)):
    def preprocess_image(file_path):
        # Orijinal görüntüyü yükle
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = (tf.cast(image, tf.float32) / 127.5) - 1  # Değerleri [-1, 1] aralığına getir

        # Bulanık görüntüyü oluştur
        blurred_image = blur_image(image)

        return blurred_image, image  # (Giriş, Hedef) çiftini döndür

    def blur_image(image):
        image = tf.expand_dims(image, axis=0)  # [H, W, C] -> [1, H, W, C]
        kernel = tf.constant([[1/16, 2/16, 1/16],
                              [2/16, 4/16, 2/16],
                              [1/16, 2/16, 1/16]],
                             dtype=tf.float32, shape=[3, 3, 1, 1])
        kernel = tf.repeat(kernel, repeats=3, axis=2)  # RGB kanalları için
        blurred = tf.nn.depthwise_conv2d(image, kernel, [1,1,1,1], 'SAME')
        blurred = tf.squeeze(blurred, axis=0)  # [1, H, W, C] -> [H, W, C]
        return blurred

    # Görüntü dosyalarını topla
    image_files = []
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, file)
                image_files.append(img_path)

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset