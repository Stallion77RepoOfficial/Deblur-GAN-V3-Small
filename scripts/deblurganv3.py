# scripts/deblurganv3.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import os
import sys

def load_image(image_path, img_size=(256, 256)):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Görüntü yüklenirken hata oluştu: {e}")
        sys.exit(1)
        
    img_array = np.array(img).astype(np.float32)
    original_shape = img_array.shape[:2]
    img_array = (img_array / 127.5) - 1.0
    img_resized = tf.image.resize(img_array, img_size)
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
    output_image = (output_resized + 1.0) * 127.5
    output_image = np.clip(output_image, 0, 255).astype('uint8')

    # Görüntüyü oluştur
    output_image = Image.fromarray(output_image)
    return output_image

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Image Deblurring Scripti. Model ve görüntü yollarını belirtin.',
        epilog='Örnek kullanım: python deblurganv3.py --model_path /path/to/model.keras --input_image /path/to/input.jpg --output_image /path/to/output.jpg'
    )
    
    parser.add_argument('--model_path', type=str, required=True, help='Deblur modelinin .keras dosya yolu')
    parser.add_argument('--input_image', type=str, required=True, help='Bulanık görüntünün dosya yolu')
    parser.add_argument('--output_image', type=str, default='deblurred_image.jpg', help='Deblurlanmış görüntünün kaydedileceği dosya yolu')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Modelin var olup olmadığını kontrol et
    if not os.path.exists(args.model_path):
        print(f"Model dosyası bulunamadı: {args.model_path}")
        sys.exit(1)
        
    # Giriş görüntüsünün var olup olmadığını kontrol et
    if not os.path.exists(args.input_image):
        print(f"Giriş görüntüsü bulunamadı: {args.input_image}")
        sys.exit(1)
    
    # Modeli yükle
    try:
        model = load_model(args.model_path, compile=False)
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        sys.exit(1)
    
    # Deblurlanmış görüntüyü elde et
    deblurred_image = deblur_image(model, args.input_image)
    
    # Çıkış görüntüsünü kaydet
    try:
        deblurred_image.save(args.output_image)
        print(f"Deblurlanmış görüntü kaydedildi: {args.output_image}")
        deblurred_image.show()
    except Exception as e:
        print(f"Çıkış görüntüsü kaydedilirken hata oluştu: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
