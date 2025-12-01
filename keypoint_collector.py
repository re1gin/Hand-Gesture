import cv2
import os
import mediapipe as mp
import numpy as np
import time

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # Untuk visualisasi titik
# =====================================================

gestures = {'rock': 'Batu', 'paper': 'Kertas', 'scissors': 'Gunting'}
DATA_DIR = 'keypoint_dataset'
num_samples = 250 # Jumlah sampel per gestur
os.makedirs(DATA_DIR, exist_ok=True)
for g in gestures: os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

cap = cv2.VideoCapture(0)
time.sleep(1)

# Visualisasi dan Instruksi
os.system('cls' if os.name == 'nt' else 'clear')
print("PENGUMPUL DATA KEYPOINT (63 FITUR) - TARGET 250/GESTURE".center(60))
available_gestures = list(gestures.keys())

while available_gestures:
    print("\nGesture belum selesai:")
    for i, g in enumerate(available_gestures, 1):
        # Hitung file .npy
        n = len([f for f in os.listdir(os.path.join(DATA_DIR, g)) if f.endswith('.npy')])
        print(f"  {i}. {gestures[g]:<10} → {n}/{num_samples} {'SELESAI' if n>=num_samples else ''}")
    
    try:
        selection = input(f"\nPilih (1-{len(available_gestures)}): ")
        if selection.lower() == 'q':
            cap.release(); cv2.destroyAllWindows(); exit()
        current = available_gestures[int(selection) - 1]
    except: 
        print("Input salah!"); continue

    print(f"\n→ SEDANG: {gestures[current].upper()}")
    time.sleep(2)
    for i in range(3,0,-1):
        ret, f = cap.read()
        if ret:
            cv2.putText(f, str(i), (280,240), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,255), 15)
            cv2.imshow('Keypoint Collector', f)
            cv2.waitKey(1000)

    # Hitung ulang count
    count = len([f for f in os.listdir(os.path.join(DATA_DIR, current)) if f.endswith('.npy')])
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Balik frame agar cermin
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        data_collected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar titik (keypoint) di layar
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # EKSTRAKSI DATA KEYPOINT 
                # Ratakan (Flatten) 21*3 koordinat (x, y, z) menjadi vektor 63 dimensi
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z]) 
                
                data_collected = np.array(keypoints, dtype=np.float32)

            cv2.putText(frame, gestures[current], (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 3)
            cv2.putText(frame, f"{count}/{num_samples}", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        else:
            cv2.putText(frame, "Tangan tidak terdeteksi", (70,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Bar perintah
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,420), (640,480), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "SPASI=Ambil 1 Foto  N=Lanjut  R=Ulangi  Q=Keluar", (10,455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow('Keypoint Collector', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            if data_collected is not False:
                # Simpan data keypoint (vektor 63 dimensi)
                path = os.path.join(DATA_DIR, current, f"{count}.npy")
                np.save(path, data_collected)
                count += 1
                print(f"→ Disimpan: {count}/{num_samples}")
                time.sleep(0.1)
            else:
                print("Tidak ada tangan terdeteksi, coba lagi.")
        
        elif k == ord('n'):
            if count >= num_samples and current in available_gestures:
                available_gestures.remove(current)
            break
        
        elif k == ord('r'):
            # Hapus semua file .npy di folder ini
            files_to_remove = [f for f in os.listdir(os.path.join(DATA_DIR, current)) if f.endswith('.npy')]
            for f in files_to_remove:
                os.remove(os.path.join(DATA_DIR, current, f))
            count = 0
            print("Folder gestur direset.")
        
        elif k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit()

    if not available_gestures:
        print("\nSELESAI SEMUA GESTURE!")
        break

cap.release()
cv2.destroyAllWindows()