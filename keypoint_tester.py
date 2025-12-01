import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# =====================================================

# Muat model yang telah dilatih
try:
    model = tf.keras.models.load_model('keypoint_model/keypoint_model.h5')
except:
    print("Error: Model 'keypoint_model/keypoint_model.h5' tidak ditemukan. Jalankan keypoint_trainer.py dulu!")
    exit()

# Mapping label
gestures = {0:'Batu', 1:'Kertas', 2:'Gunting'}
cap = cv2.VideoCapture(0)

print("Sistem Pengenalan Gestur Keypoint Aktif...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Balik frame agar cermin
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    label = "Menunggu Tangan..." # Label default

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Ekstraksi Keypoint (63 Fitur)
            keypoints = []
            for lm in hand_landmarks.landmark:
                # Kita hanya perlu x, y, z yang sudah dinormalisasi oleh MediaPipe
                keypoints.extend([lm.x, lm.y, lm.z])
            
            input_data = np.array(keypoints, dtype=np.float32)
            
            # 2. Prediksi Model
            # Model membutuhkan input berdimensi (1, 63)
            pred = model.predict(input_data[np.newaxis,...], verbose=0)
            
            # 3. Proses Hasil
            pred_class = np.argmax(pred)
            confidence = pred[0][pred_class]
            label = f"{gestures[pred_class]} ({confidence*100:.2f}%)"
            
            # 4. Visualisasi (Gambar titik dan garis)
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Titik hijau
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2) # Garis merah
            )

    # Tampilkan hasil prediksi di frame
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow('Real-Time Keypoint Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()