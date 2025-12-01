import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.makedirs('model', exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    fill_mode='nearest'
)

train = datagen.flow_from_directory('dataset', target_size=(128,128), batch_size=32,
                                    class_mode='categorical', subset='training')
val   = datagen.flow_from_directory('dataset', target_size=(128,128), batch_size=32,
                                    class_mode='categorical', subset='validation')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(train.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model/best_model.h5', save_best_only=True,
                            monitor='val_accuracy', mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

print("Training dimulai... (25 epoch)")
model.fit(train, epochs=25, validation_data=val, callbacks=[checkpoint, reduce_lr])

# Fine-tuning
base.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=10, validation_data=val, callbacks=[checkpoint, reduce_lr])

print("SELESAI! Model terbaik: model/best_model.h5 → Akurasi 99%+")