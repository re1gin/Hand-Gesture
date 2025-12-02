import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================== Pengaturan ==================
DATA_DIR = 'keypoint_dataset_v4'
MODEL_PATH = 'keypoint_model_v7/youtube_controller_mlp_v7.h5'
os.makedirs('keypoint_model_v7', exist_ok=True)

gestures = {
    'open_palm': 'Netral',
    'fist': 'Play/Pause (Space)',
    'thumb_up': 'Forward (Right)',
    'thumb_down': 'Rewind (Left)',
    'index_up': 'Vol Up (Up Arrow)',
    'index_down': 'Vol Down (Down Arrow)',
    'c_shape': 'Subtitle (C)',
    'ok_sign': 'Click',
    'two_fingers_up': 'Cursor',
    'three_fingers_up': 'Mute (M)',
    'four_fingers': 'Fullscreen (F)',
    'index_side_90': 'Teater Mode (T)',
    'two_fingers_side_90': 'Open YT',
    'two_fingers_side_back': 'Enter',
    'three_fingers_side_90': 'Close Tab',
    'pinky_up': 'Esc',
    'L_shape': 'Next (Shift+N)',
    'gun_shape': 'Previous (Shift+P)',
    'two_fingers_up_other': 'Scroll Up',
    'two_fingers_down_other': 'Scroll Down',
    'thumb_left': 'Back Page (Alt+Left)',
    'thumb_right': 'Forward Page (Alt+Right)',
}
label_map = {name: i for i, name in enumerate(gestures.keys())} 
num_classes = len(gestures)
# =================================================

# 1. MUAT DATA
X = [] 
y = [] 

print(f"Memuat {num_classes} kelas gestur dari {DATA_DIR}...")
for gesture_name, label_int in label_map.items():
    path = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(path): 
        print(f"⚠️ Peringatan: Folder data '{gesture_name}' tidak ditemukan di {DATA_DIR}. Lewati.")
        continue
        
    for file_name in os.listdir(path):
        if file_name.endswith('.npy'):
            data = np.load(os.path.join(path, file_name))
            if data.shape[0] == 63: 
                X.append(data)
                y.append(label_int)
print(f"Total data dimuat: {len(X)} sampel.")

if len(X) == 0:
    print("\n❌ Gagal memuat data. Pastikan DATA_DIR benar dan sudah ada file .npy.")
    exit()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
y = to_categorical(y, num_classes=num_classes)

# 2. SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
print(f"Data Training: {len(X_train)}, Data Validasi: {len(X_val)}")

# 3. DEFINISI MODEL (MLP)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
              
model.summary()

# 4. CALLBACKS
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True,
                             monitor='val_accuracy', mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# 5. PELATIHAN
print("\nTraining dimulai...")
history = model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop]
)

print("\nPelatihan Selesai! Model terbaik disimpan di:", MODEL_PATH)
best_acc = max(history.history['val_accuracy'])
print(f"Akurasi Validasi Tertinggi: {best_acc*100:.2f}%")