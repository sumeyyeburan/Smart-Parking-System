import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ── Ayarlar ───────────────────────────────────────────────────────────────────
BASE_DIR   = "data"
LABEL_FILE = os.path.join(BASE_DIR, "sunny.txt")
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 15
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    print("\nVeri yükleniyor...")
    df = pd.read_csv(LABEL_FILE, sep=' ', header=None, names=['img_path', 'label'])
    df['full_path'] = df['img_path'].apply(lambda x: os.path.join(BASE_DIR, x))
    df['label'] = df['label'].astype(str)

    df = df[df['full_path'].apply(os.path.exists)]
    print(f"Toplam görüntü: {len(df)}")
    print(f"Dolu (1): {sum(df['label']=='1')}")
    print(f"Boş  (0): {sum(df['label']=='0')}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ── Generator'lar ──
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='full_path', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='full_path', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )

    # ── Model — MobileNetV2 Transfer Learning ──
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # ── Eğitim ──
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]

    print("\nEğitim başlıyor...")
    start = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\nEğitim tamamlandı! Süre: {(time.time()-start)/60:.1f} dakika")

    # ── Kaydet ──
    os.makedirs("models", exist_ok=True)
    model.save("models/parking_cnn.h5")
    print("Model kaydedildi: models/parking_cnn.h5")

if __name__ == "__main__":
    main()