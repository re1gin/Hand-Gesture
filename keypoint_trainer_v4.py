import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================== Pengaturan ==================
# CATATAN: Karena Anda menambahkan 4 gestur baru, ini secara fungsional adalah V6 (20 Kelas)
DATA_DIR = 'keypoint_dataset_v4' # Tetap menggunakan folder data V4 (sesuai permintaan user)
MODEL_PATH = 'keypoint_model_v6/youtube_controller_mlp_v6.h5' # <-- Path model V6 (20 Kelas)
os.makedirs('keypoint_model_v6', exist_ok=True) # Buat folder model V6

# Gestur FINAL (20 Kelas Total - Urutan harus sesuai dengan saat koleksi data)
gestures = {
    # ID 0 - 7
    'open_palm': 'Netral',          # ID 0
    'fist': 'Play/Pause',           # ID 1
    'thumb_up': 'Rewind',           # ID 2
    'thumb_down': 'Forward',        # ID 3
    'index_up': 'Vol Up',           # ID 4
    'index_down': 'Vol Down',       # ID 5
    'c_shape': 'Subtitle',          # ID 6: Aksi diubah (Subtitle)
    'ok_sign': 'Click',             # ID 7
    
    # ID 8 - 15
    'two_fingers_up': 'Cursor',      # ID 8 (Dinamis: Mouse Move)
    'three_fingers_up': 'Mute',      # ID 9: Aksi diubah (Mute Statis)
    'four_fingers': 'Fullscreen',    # ID 10
    'index_side_90': 'Teater Mode',  # ID 11: Nama aksi disesuaikan
    'two_fingers_side_90': 'Open YT',# ID 12
    'two_fingers_side_back': 'Enter',# ID 13 
    'three_fingers_side_90': 'Close Tab',# ID 14
    'pinky_up': 'Esc',               # ID 15

    # ID 16 - 19 (GESTUR BARU)
    'L_shape': 'Next (Shift+N)',         # ID 16
    'gun_shape': 'Previous (Shift+P)',   # ID 17
    'two_fingers_up_other': 'Scroll Up', # ID 18: Gestur Scroll Up Statis
    'two_fingers_down_other': 'Scroll Down', # ID 19: Gestur Scroll Down Statis
}
label_map = {name: i for i, name in enumerate(gestures.keys())} 
num_classes = len(gestures) # Jumlah kelas sekarang 20
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
            # Keypoint dinormalisasi memiliki dimensi 63 (21 landmark * 3 koordinat)
            if data.shape[0] == 63: 
                X.append(data)
                y.append(label_int)
print(f"Total data dimuat: {len(X)} sampel.")

# Jika tidak ada data yang dimuat, hentikan proses
if len(X) == 0:
    print("\n❌ Gagal memuat data. Pastikan DATA_DIR benar dan sudah ada file .npy.")
    exit()

# Konversi ke NumPy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
# Konversi label integer ke one-hot encoding
y = to_categorical(y, num_classes=num_classes) # Menggunakan num_classes=20

# 2. SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
print(f"Data Training: {len(X_train)}, Data Validasi: {len(X_val)}")

# 3. DEFINISI MODEL (MLP)
# Output layer harus sesuai dengan jumlah kelas baru (20)
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
              
model.summary() # Tampilkan ringkasan model

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