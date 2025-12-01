import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================== Pengaturan ==================
DATA_DIR = 'keypoint_dataset'
MODEL_PATH = 'keypoint_model/keypoint_model.h5'
os.makedirs('keypoint_model', exist_ok=True)

gestures = {'rock': 'Batu', 'paper': 'Kertas', 'scissors': 'Gunting'}
label_map = {name: i for i, name in enumerate(gestures.keys())} # rock: 0, paper: 1, scissors: 2
# =================================================

# 1. MUAT DATA
X = [] # Data fitur (keypoints)
y = [] # Data label

print("Memuat data keypoint...")
for gesture_name, label_int in label_map.items():
    path = os.path.join(DATA_DIR, gesture_name)
    for file_name in os.listdir(path):
        if file_name.endswith('.npy'):
            data = np.load(os.path.join(path, file_name))
            X.append(data)
            y.append(label_int)
print(f"Total data dimuat: {len(X)} sampel.")

# Konversi ke NumPy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
y = to_categorical(y, num_classes=len(gestures)) # One-Hot Encoding

# 2. SPLIT DATA (Train dan Validation)
# Data 90% Training, 10% Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
print(f"Data Training: {len(X_train)}, Data Validasi: {len(X_val)}")

# 3. DEFINISI MODEL (MLP - Fully Connected Network)
# Input shape adalah 63 (21 landmarks * 3 koordinat)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 4. CALLBACKS
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True,
                             monitor='val_accuracy', mode='max', verbose=1)
# Hentikan pelatihan jika akurasi validasi tidak meningkat setelah 10 epoch
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 5. PELATIHAN
print("\nTraining dimulai...")
history = model.fit(
    X_train, y_train,
    epochs=50, # Cukup 50 epoch, model keypoint cepat konvergen
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop]
)

print("\nPelatihan Selesai! Model terbaik disimpan di:", MODEL_PATH)

# Opsional: Tampilkan akurasi tertinggi
best_acc = max(history.history['val_accuracy'])
print(f"Akurasi Validasi Tertinggi: {best_acc*100:.2f}%")