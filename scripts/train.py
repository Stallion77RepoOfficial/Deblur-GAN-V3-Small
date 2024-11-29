# scripts/train.py

import sys
import os

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import yaml
import argparse
from models.unet import build_optimized_unet
from dataset.dataset import load_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import datetime
from tensorflow.keras import mixed_precision

def load_config(config_path='/Users/berkegulacar/Downloads/deblurganv3/config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(config):
    # Mixed precision'ı etkinleştir (isteğe bağlı)
    if config['training'].get('mixed_precision', False):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed Precision etkinleştirildi.")
    else:
        print("Mixed Precision devre dışı bırakıldı.")

    # Metal API kullanımı için TensorFlow yapılandırması
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU kullanılıyor")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU bulunamadı, CPU kullanılıyor")

    # Modeli oluştur
    model = build_optimized_unet(input_shape=tuple(config['models']['input_shape']))
    
    # learning_rate'ın float olduğundan emin olun
    learning_rate = config['training']['learning_rate']
    if isinstance(learning_rate, str):
        try:
            learning_rate = float(learning_rate)
            config['training']['learning_rate'] = learning_rate
            print(f"learning_rate string'den float'a dönüştürüldü: {learning_rate}")
        except ValueError:
            print(f"learning_rate değeri dönüştürülemez: {learning_rate}")
            sys.exit(1)
    elif not isinstance(learning_rate, float):
        print(f"learning_rate beklenen tipte değil: {type(learning_rate)}")
        sys.exit(1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )

    # Veri setini yükle
    train_dataset, val_dataset = load_dataset(config)

    # Callback'ler
    checkpoint = ModelCheckpoint(
        filepath=config['training']['model_save_path'],
        save_best_only=config['training']['save_best_only'],
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['lr_reduce_factor'],
        patience=config['training']['lr_reduce_patience'],
        min_lr=config['training']['min_lr'],
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stop_patience'],
        restore_best_weights=True,
        verbose=1
    )
    log_dir = config['logging']['tensorboard_log_dir'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoint, reduce_lr, early_stop, tensorboard_callback]

    # Model özeti
    model.summary()

    # Eğitim
    model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Son modelin kaydedilmesi
    model.save(config['training']['final_model_save_path'])
    print(f"Son model kaydedildi: {config['training']['final_model_save_path']}")

def main():
    parser = argparse.ArgumentParser(description='DeblurganV3 Eğitim Scripti')
    parser.add_argument('--config', type=str, default='/Users/berkegulacar/Downloads/deblurganv3/config/config.yaml', help='Konfigürasyon dosyasının yolu')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Konfigürasyon dosyası bulunamadı: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    train(config)

if __name__ == "__main__":
    main()
