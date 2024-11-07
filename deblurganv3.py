# usage.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32)
    original_shape = img_array.shape[:2]
    img_array = (img_array / 127.5) - 1
    img_resized = tf.image.resize(img_array, (256, 256))
    return img_resized, original_shape

def deblur_image(model, image_path):
    input_image, original_shape = load_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)

    # Tahmin yap
    output = model.predict(input_image)

    # Çıkışı orijinal boyuta ölçeklendir
    output_resized = tf.image.resize(output[0], original_shape)

    # TensorFlow tensörünü NumPy array'e dönüştür
    output_resized = output_resized.numpy()

    # Değerleri [0, 255] aralığına getir ve uint8 tipine dönüştür
    output_image = (output_resized + 1) * 127.5
    output_image = np.clip(output_image, 0, 255).astype('uint8')

    # Görüntüyü oluştur
    output_image = Image.fromarray(output_image)
    return output_image

if __name__ == "__main__":
    model = load_model('/Users/berkegulacar/Downloads/deblurganv2/small_model.h5', compile=False)
    deblurred_image = deblur_image(model, '/Users/berkegulacar/Downloads/blur_image.jpg')
    deblurred_image.save('deblurred_image.jpg')
    deblurred_image.show()