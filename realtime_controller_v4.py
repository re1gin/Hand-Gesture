import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import pyautogui
import webbrowser 

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# =====================================================

# FUNGSI NORMALISASI KEYPOINT (WAJIB SAMA DENGAN COLLECTOR & TRAINER)
def normalize_keypoints(hand_landmarks):
    all_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]) 
    wrist = all_coords[0, :2] 
    translated_coords = all_coords[:, :2] - wrist 
    dist_ref = np.linalg.norm(translated_coords[9] - translated_coords[0]) 
    if dist_ref < 1e-6: dist_ref = 1.0 
    normalized_coords_xy = translated_coords / dist_ref

    keypoints = []
    for i in range(21):
        keypoints.extend([normalized_coords_xy[i, 0], normalized_coords_xy[i, 1], all_coords[i, 2]])
    return np.array(keypoints, dtype=np.float32)

# =====================================================

# Muat model yang telah dilatih
MODEL_PATH = 'keypoint_model_v4/youtube_controller_mlp.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    print(f"Error: Model '{MODEL_PATH}' tidak ditemukan. Latih ulang model dengan 16 kelas!")
    exit()

# Mapping label BARU (16 Kelas)
gestures = {
    0:'OPEN_PALM (Default)', 1:'FIST (Play/Pause)', 2:'THUMB_UP (Rewind)', 
    3:'THUMB_DOWN (Forward)', 4:'INDEX_UP (Vol Up)', 5:'INDEX_DOWN (Vol Down)',
    6:'C_SHAPE (Mute)', 7:'OK_SIGN (Enter)', 8:'TWO_FINGERS_UP (Cursor)', 
    9:'THREE_FINGERS_UP (Scroll)', 10:'FOUR_FINGERS (Fullscreen)', 
    11:'INDEX_SIDE_90 (Tab)', 12:'TWO_FINGERS_SIDE_90 (Open YT)', 
    13:'THREE_FINGERS_SIDE_90 (Close Tab)', 14:'PINKY_UP (Esc)',
    15:'SWIPE_UP_DOWN (Scroll Logic)' # Dipetakan ke ID 9/15 untuk Scroll
}
cap = cv2.VideoCapture(0)

# Variabel Kontrol UX/Aksi
PREDICTION_HISTORY = []
HISTORY_SIZE = 5         
MIN_CONFIDENCE = 0.85    
LAST_ACTIVATION = time.time()
ACTIVATION_COOLDOWN = 0.5 

LAST_OPEN_YT = time.time()
OPEN_YT_COOLDOWN = 30.0 

# Variabel Kursor dan Scroll (Logika Dinamis)
PREV_HAND_Y = None
PREV_HAND_X = None
SCROLL_SENSITIVITY = 15 # Kepekaan scroll
CURSOR_SENSITIVITY = 2.5 # Kepekaan kursor

def activate_gesture_action(gesture_id):
    """Memetakan ID gestur ke aksi PyAutoGUI dan mengontrol Cooldown."""
    global LAST_ACTIVATION, LAST_OPEN_YT
    
    # Cooldown standar untuk aksi cepat (kecuali aksi kursor dan scroll)
    if time.time() - LAST_ACTIVATION < ACTIVATION_COOLDOWN and gesture_id not in [8, 9, 15, 12]:
        return
    
    action_map = {
        # Aksi Kontrol Media
        1: lambda: pyautogui.press('space'),       # FIST (Play/Pause)
        2: lambda: pyautogui.press('left'),        # THUMB_UP (Rewind)
        3: lambda: pyautogui.press('right'),       # THUMB_DOWN (Forward)
        4: lambda: pyautogui.press('up'),          # INDEX_UP (Vol Up)
        5: lambda: pyautogui.press('down'),        # INDEX_DOWN (Vol Down)
        6: lambda: pyautogui.press('m'),           # C_SHAPE (Mute)
        7: lambda: pyautogui.press('enter'),       # OK_SIGN (Enter/Pilih)
        10: lambda: pyautogui.press('f'),          # FOUR_FINGERS (Fullscreen)
        11: lambda: pyautogui.press('tab'),         # INDEX_SIDE_90 (Tab)
        13: lambda: pyautogui.hotkey('ctrl', 'w'),  # THREE_FINGERS_SIDE_90 (Close Tab)
        14: lambda: pyautogui.press('esc')          # PINKY_UP (Esc)
    }
    
    # Aksi Khusus Buka YouTube (ID 12)
    if gesture_id == 12:
        if time.time() - LAST_OPEN_YT >= OPEN_YT_COOLDOWN:
            webbrowser.open('https://www.youtube.com', new=2)
            print(f"✅ AKSI AKTIF: {gestures[gesture_id]} | COOLDOWN: {OPEN_YT_COOLDOWN}s")
            LAST_OPEN_YT = time.time() 
        else:
            print(f"⏳ Cooldown Buka YouTube masih aktif! ({round(OPEN_YT_COOLDOWN - (time.time() - LAST_OPEN_YT))}s tersisa)")
            return 
    
    action = action_map.get(gesture_id)
    if action:
        action()
        print(f"✅ AKSI AKTIF: {gestures[gesture_id]}")
        LAST_ACTIVATION = time.time() # Reset Cooldown Standar

# =========================================================
# FUNGSI DINAMIS KHUSUS (Kursor dan Scroll)
# =========================================================
def handle_cursor_and_scroll(frame, hand_landmarks, pred_class):
    global PREV_HAND_Y, PREV_HAND_X
    
    # Mengambil koordinat pergelangan tangan (landmark 0) dinormalisasi ke ukuran frame
    wrist_lm = hand_landmarks.landmark[0]
    hand_y = wrist_lm.y * frame.shape[0]
    hand_x = wrist_lm.x * frame.shape[1]

    # Inisialisasi posisi awal
    if PREV_HAND_Y is None:
        PREV_HAND_Y = hand_y
        PREV_HAND_X = hand_x
        return
    
    if pred_class in [8]: # Gestur Kursor (ID 8: TWO_FINGERS_UP)
        # Hitung perubahan relatif
        dx = (hand_x - PREV_HAND_X) * CURSOR_SENSITIVITY
        dy = (hand_y - PREV_HAND_Y) * CURSOR_SENSITIVITY
        
        # Pindahkan kursor (pyautogui.moveRel)
        # Catatan: Tangan yang di-flip membuat X terbalik, dikalikan -1
        pyautogui.move(int(dx * -1), int(dy))
        
        # Tampilkan status kursor di frame
        cv2.putText(frame, "CURSOR ACTIVE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        
    elif pred_class in [9, 15]: # Gestur Scroll (ID 9: THREE_FINGERS_UP, ID 15: SWIPE_UP_DOWN)
        # Hitung perubahan vertikal
        dy = hand_y - PREV_HAND_Y
        
        if abs(dy) > SCROLL_SENSITIVITY:
            scroll_amount = 100 # Scroll 100 unit
            if dy < 0: # Tangan bergerak ke atas (Scrolling ke atas di layar)
                pyautogui.scroll(scroll_amount)
                cv2.putText(frame, "SCROLL UP", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else: # Tangan bergerak ke bawah (Scrolling ke bawah di layar)
                pyautogui.scroll(-scroll_amount)
                cv2.putText(frame, "SCROLL DOWN", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Perbarui posisi sebelumnya hanya jika gestur dinamis aktif, 
    # atau reset jika gestur netral (ID 0) terdeteksi.
    if pred_class in [8, 9, 15]:
        PREV_HAND_Y = hand_y
        PREV_HAND_X = hand_x
    elif pred_class == 0:
        PREV_HAND_Y = None
        PREV_HAND_X = None
# =========================================================


print("🎥 Sistem Navigasi YouTube Gestur V4 Aktif...")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    label = "Menunggu Tangan..."
    
    if not results.multi_hand_landmarks:
        PREDICTION_HISTORY = []
        PREV_HAND_Y = None # Reset scroll/cursor state
        PREV_HAND_X = None
        cv2.putText(frame, "❌ Tangan tidak terdeteksi", (70,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            input_data = normalize_keypoints(hand_landmarks)
            
            if input_data.ndim == 1: input_data = input_data[np.newaxis, ...]
            
            pred = model.predict(input_data, verbose=0)
            
            # Stabilitas Prediksi
            PREDICTION_HISTORY.append(pred)
            if len(PREDICTION_HISTORY) > HISTORY_SIZE:
                PREDICTION_HISTORY.pop(0)
            
            avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
            
            # Pastikan avg_pred memiliki dimensi yang benar sebelum argmax
            if avg_pred.ndim > 1:
                pred_class = np.argmax(avg_pred[0])
                confidence = avg_pred[0][pred_class]
            else:
                 # Fallback jika model output tidak sesuai harapan
                pred_class = 0 
                confidence = 0.0
            
            color = (0, 0, 255) 
            
            # Pemicu Aksi Statis
            if confidence >= MIN_CONFIDENCE:
                
                # Panggil logika Kursor/Scroll (khusus untuk ID 8, 9, 15)
                if pred_class in [8, 9, 15]:
                    handle_cursor_and_scroll(frame, hand_landmarks, pred_class)
                    color = (255, 100, 255) # Ungu untuk dinamis
                
                # Panggil logika Statis (selain ID 0, 8, 9, 15)
                elif pred_class != 0:
                    activate_gesture_action(pred_class)
                    color = (0, 255, 0) 
                
                # Gestur Default
                elif pred_class == 0:
                     color = (255, 255, 0)
                
                label = f"TERDETEKSI: {gestures[pred_class]} ({confidence*100:.1f}%)"
                
            else:
                color = (0, 165, 255) 
                label = f"Menganalisis: {gestures.get(pred_class, 'Unknown')} ({confidence*100:.1f}%)"
            
            # Visualisasi Keypoint
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Tampilkan Feedback Gestur (UX)
            if 'avg_pred' in locals() and avg_pred.ndim > 1:
                top_3_indices = np.argsort(avg_pred[0])[::-1][:3]
                cv2.putText(frame, "Top 3 Prediksi:", (w-250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                for i, idx in enumerate(top_3_indices):
                    text_label = f"{gestures.get(idx, 'Unknown')}: {avg_pred[0][idx]*100:.1f}%"
                    cv2.putText(frame, text_label, (w-250, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan Label Status
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow('YouTube Gesture Controller V4', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()