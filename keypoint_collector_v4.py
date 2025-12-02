import cv2
import os
import mediapipe as mp
import numpy as np
import time

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# =====================================================

gestures = {
    # ID 0 - 7
    'open_palm': 'Netral',          # ID 0
    'fist': 'Play/Pause',           # ID 1
    'thumb_up': 'Rewind',           # ID 2
    'thumb_down': 'Forward',        # ID 3
    'index_up': 'Vol Up',           # ID 4
    'index_down': 'Vol Down',       # ID 5
    'c_shape': 'Subtitle',          # ID 6
    'ok_sign': 'Click',             # ID 7
    
    # ID 8 - 15
    'two_fingers_up': 'Cursor',      # ID 8 (Dinamis)
    'three_fingers_up': 'Mute',    # ID 9
    'four_fingers': 'Fullscreen',    # ID 10
    'index_side_90': 'Teater Mode',  # ID 11
    'two_fingers_side_90': 'Open YT',# ID 12
    'two_fingers_side_back': 'Enter',# ID 13 
    'three_fingers_side_90': 'Close Tab',# ID 14
    'pinky_up': 'Esc',               # ID 15

    # ID 16 & 17 (GESTUR BARU)
    'L_shape': 'Next (Shift+N)',     # ID 16: Jempol & Telunjuk (Lurus 90 derajat)
    'gun_shape': 'Previous (Shift+P)', # ID 17: Jempol & Telunjuk (Membentuk Pistol)
    
    'two_fingers_up_other': 'Scroll Up',
    'two_fingers_down_other': 'Scroll Down',
}
DATA_DIR = 'keypoint_dataset_v4'
num_samples = 250

# =====================================================
# FUNGSI NORMALISASI KEYPOINT (Invarian Posisi & Skala)
# =====================================================
def normalize_keypoints(hand_landmarks):
    """Menghasilkan vektor 63 dimensi (x, y, z) yang dinormalisasi."""
    all_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]) 
    
    # 1. Translasi (Posisi) - Relatif terhadap pergelangan tangan (index 0)
    wrist = all_coords[0, :2] 
    translated_coords = all_coords[:, :2] - wrist 
    
    # 2. Skala (Ukuran) - Normalisasi berdasarkan jarak 0 ke 9 (pergelangan ke pangkal jari tengah)
    dist_ref = np.linalg.norm(translated_coords[9] - translated_coords[0]) 
    
    if dist_ref < 1e-6:
        dist_ref = 1.0 

    normalized_coords_xy = translated_coords / dist_ref

    # 3. Kumpulkan kembali (Normalisasi x,y + z mentah)
    keypoints = []
    for i in range(21):
        keypoints.extend([normalized_coords_xy[i, 0], normalized_coords_xy[i, 1], all_coords[i, 2]])

    return np.array(keypoints, dtype=np.float32)
# =====================================================

# Persiapan Folder
os.makedirs(DATA_DIR, exist_ok=True)
for g in gestures: os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

cap = cv2.VideoCapture(0)
time.sleep(1)

# Visualisasi dan Instruksi
os.system('cls' if os.name == 'nt' else 'clear')
print("PENGUMPUL DATA KEYPOINT (18 GESTUR DINORMALISASI) - TARGET 250/GESTURE".center(80))
available_gestures = list(gestures.keys())

# Variabel kontrol untuk keluar dari loop utama
quit_program = False 

while available_gestures and not quit_program:
    print("\n📦 Gesture belum selesai:")
    for i, g in enumerate(available_gestures, 1):
        n = len([f for f in os.listdir(os.path.join(DATA_DIR, g)) if f.endswith('.npy')])
        # Menampilkan ID yang benar (i-1)
        print(f" {i-1:<2}. {g:<20} -> {gestures[g]:<15} → {n}/{num_samples} {'✅ SELESAI' if n>=num_samples else ''}")
    
    try:
        selection = input(f"\nPilih ID (0-{len(available_gestures)-1}) atau 'q' untuk keluar: ")
        if selection.lower() == 'q':
            quit_program = True
            break
        # Mengambil gestur berdasarkan ID (index)
        current = available_gestures[int(selection)]
    except: 
        print("Input salah! Masukkan ID angka yang valid atau 'q'."); continue

    if quit_program:
        break

    print(f"\n▶️ SEDANG: {gestures[current].upper()} - Siapkan Tangan Anda")
    time.sleep(1)
    
    # Hitung mundur
    for i in range(3,0,-1):
        ret, f = cap.read()
        if ret:
            f = cv2.flip(f, 1)
            cv2.putText(f, str(i), (280,240), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,255), 15)
            cv2.imshow('Keypoint Collector', f)
            if cv2.waitKey(1000) & 0xFF == ord('q'): 
                quit_program = True
                break
    
    if quit_program:
        break

    count = len([f for f in os.listdir(os.path.join(DATA_DIR, current)) if f.endswith('.npy')])
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        data_collected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data_collected = normalize_keypoints(hand_landmarks)

            cv2.putText(frame, gestures[current], (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 3)
            cv2.putText(frame, f"Sisa: {num_samples-count}", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        else:
            cv2.putText(frame, "❌ Tangan tidak terdeteksi", (70,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Bar perintah
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,420), (640,480), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "SPASI=Ambil Data  N=Selesai Gestur  R=Reset  Q=Keluar", (10,455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow('Keypoint Collector', frame)

        k = cv2.waitKey(1) & 0xFF
        
        # Penanganan Tombol 'Q' yang terpusat
        if k == ord('q'):
            quit_program = True
            break
        
        # Penanganan Tombol Lainnya
        if k == ord(' '):
            if data_collected is not False and count < num_samples:
                path = os.path.join(DATA_DIR, current, f"{count}.npy")
                np.save(path, data_collected)
                count += 1
                print(f"→ Disimpan ({gestures[current]}): {count}/{num_samples}")
                time.sleep(0.1)
            elif count >= num_samples:
                print("Kuota sampel sudah terpenuhi! Tekan 'N' untuk lanjut.")
            else:
                print("Tidak ada tangan terdeteksi, coba lagi.")
        
        elif k == ord('n'):
            if count >= num_samples:
                if current in available_gestures:
                    available_gestures.remove(current)
                break
            else:
                print(f"Sampel kurang, butuh {num_samples-count} lagi.")
        
        elif k == ord('r'):
            files_to_remove = [f for f in os.listdir(os.path.join(DATA_DIR, current)) if f.endswith('.npy')]
            for f in files_to_remove:
                os.remove(os.path.join(DATA_DIR, current, f))
            count = 0
            print("Folder gestur direset.")
            
    if quit_program:
        break

if not available_gestures and not quit_program:
    print("\n🎉 SELESAI SEMUA GESTURE! Semua data siap untuk training.")

# Pembersihan di akhir
cap.release()
cv2.destroyAllWindows()