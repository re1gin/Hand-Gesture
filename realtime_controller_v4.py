import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import pyautogui
import webbrowser 

# ================== Setup UI Parameters ==================
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
FONT_THICKNESS = 1
WHITE = (255, 255, 255)
RED = (0, 0, 255)
# =========================================================

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# =====================================================

# FUNGSI NORMALISASI KEYPOINT (WAJIB SAMA DENGAN COLLECTOR & TRAINER)
def normalize_keypoints(hand_landmarks):
    """Menghasilkan vektor 63 dimensi yang dinormalisasi."""
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

MODEL_PATH = 'keypoint_model_v7/youtube_controller_mlp_v7.h5'
NUM_CLASSES = 22

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    if model.output_shape[1] != NUM_CLASSES: 
        print(f"Error: Model memiliki {model.output_shape[1]} kelas, harusnya {NUM_CLASSES} (V7).")
        exit()
except Exception as e:
    print(f"Error: Model '{MODEL_PATH}' tidak ditemukan atau gagal dimuat. {e}")
    print("Pastikan Anda sudah melatih model V7 (22 kelas)!")
    exit()

gestures = {
    0:'OPEN_PALM (Default)', 1:'FIST (Play/Pause)', 2:'THUMB_UP (Forward)', 
    3:'THUMB_DOWN (Rewind)', 4:'INDEX_UP (Vol Up)', 5:'INDEX_DOWN (Vol Down)',
    6:'C_SHAPE (Subtitle)', 
    7:'OK_SIGN (Click)', 
    
    8:'TWO_FINGERS_UP (Cursor)', 
    9:'THREE_FINGERS_UP (Mute)', 
    10:'FOUR_FINGERS (Fullscreen)', 11:'INDEX_SIDE_90 (Teater Mode)', 
    12:'TWO_FINGERS_SIDE_90 (Open YT)', 
    13:'TWO_FINGERS_SIDE_BACK (Enter)', 
    14:'THREE_FINGERS_SIDE_90 (Close Tab)',
    15:'PINKY_UP (Esc)',
    
    16:'L_SHAPE (Next)', 
    17:'GUN_SHAPE (Previous)', 
    18:'SCROLL_UP (Static)', 
    19:'SCROLL_DOWN (Static)',
    
    20:'THUMB_LEFT (Back Page)',
    21:'THUMB_RIGHT (Forward Page)',
}

def get_gesture_name(gesture_string):
    return gesture_string.split(' (')[0]

cap = cv2.VideoCapture(0)

# Variabel Kontrol UX/Aksi
PREDICTION_HISTORY = []
HISTORY_SIZE = 5 
MIN_CONFIDENCE = 0.85 
LAST_ACTIVATION = time.time()
ACTIVATION_COOLDOWN = 1.0 
PAGE_NAV_COOLDOWN = 2.0

LAST_OPEN_YT = time.time()
OPEN_YT_COOLDOWN = 10.0 

# Variabel Kursor dan Scroll
PREV_HAND_Y = None
PREV_HAND_X = None
GLOBAL_SMOOTH_DX = 0
GLOBAL_SMOOTH_DY = 0
CURSOR_SENSITIVITY = 3.8
SCROLL_AMOUNT = 200 

def activate_gesture_action(gesture_id):
    """Memetakan ID gestur ke aksi PyAutoGUI dan mengontrol Cooldown."""
    global LAST_ACTIVATION, LAST_OPEN_YT
    
    cooldown_duration = ACTIVATION_COOLDOWN
    if gesture_id in [20, 21]:
        cooldown_duration = PAGE_NAV_COOLDOWN
    
    if time.time() - LAST_ACTIVATION < cooldown_duration and gesture_id not in [8, 12]:
        return
    
    action_map = {
        1: lambda: pyautogui.press('space'), 
        2: lambda: pyautogui.press('left'), 
        3: lambda: pyautogui.press('right'), 
        4: lambda: pyautogui.press('up'), 
        5: lambda: pyautogui.press('down'), 
        6: lambda: pyautogui.press('c'), 
        7: lambda: pyautogui.click(), 
        9: lambda: pyautogui.press('m'), 
        10: lambda: pyautogui.press('f'), 
        11: lambda: pyautogui.press('t'), 
        13: lambda: pyautogui.press('enter'),
        14: lambda: pyautogui.hotkey('ctrl', 'w'),
        15: lambda: pyautogui.press('esc'), 
        16: lambda: pyautogui.hotkey('shift', 'n'),
        17: lambda: pyautogui.hotkey('shift', 'p'),
        18: lambda: pyautogui.scroll(SCROLL_AMOUNT),
        19: lambda: pyautogui.scroll(-SCROLL_AMOUNT),
        
        20: lambda: pyautogui.hotkey('alt', 'left'),
        21: lambda: pyautogui.hotkey('alt', 'right'),
    }
    
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
        # Mengatur LAST_ACTIVATION dengan durasi cooldown yang dipilih
        if gesture_id in [20, 21]:
            print(f"✅ AKSI AKTIF: {gestures[gesture_id]} | COOLDOWN: {PAGE_NAV_COOLDOWN}s") 
        else:
            print(f"✅ AKSI AKTIF: {gestures[gesture_id]}") 
            
        LAST_ACTIVATION = time.time() 

# =========================================================
# FUNGSI DINAMIS KHUSUS (Hanya Kursor - ID 8)
# =========================================================

def handle_cursor_movement(frame, hand_landmarks, pred_class):
    global PREV_HAND_Y, PREV_HAND_X
    global GLOBAL_SMOOTH_DX, GLOBAL_SMOOTH_DY

    wrist_lm = hand_landmarks.landmark[0]
    hand_y = wrist_lm.y * frame.shape[0]
    hand_x = wrist_lm.x * frame.shape[1]

    if PREV_HAND_X is None:
        PREV_HAND_X = hand_x
        PREV_HAND_Y = hand_y
        return

    if pred_class == 8:

        dx_raw = hand_x - PREV_HAND_X
        dy_raw = hand_y - PREV_HAND_Y

        # Deadzone
        DEADZONE = 4
        if abs(dx_raw) < DEADZONE: dx_raw = 0
        if abs(dy_raw) < DEADZONE: dy_raw = 0

        # Exponential smoothing
        SMOOTHING = 0.12
        GLOBAL_SMOOTH_DX = GLOBAL_SMOOTH_DX * (1 - SMOOTHING) + dx_raw * SMOOTHING
        GLOBAL_SMOOTH_DY = GLOBAL_SMOOTH_DY * (1 - SMOOTHING) + dy_raw * SMOOTHING

        # Soft acceleration
        speed = np.sqrt(dx_raw**2 + dy_raw**2)
        accel = 1 + min(speed * 0.010, 0.8)

        # Wide movement scaling
        CURSOR_SENSITIVITY = 5.5
        dx = GLOBAL_SMOOTH_DX * CURSOR_SENSITIVITY * accel
        dy = GLOBAL_SMOOTH_DY * CURSOR_SENSITIVITY * accel

        pyautogui.move(int(-dx), int(dy))

        cv2.putText(frame, "CURSOR ACTIVE (Smooth + Wide + Stable)", (20, 70),
                    FONT_FACE, FONT_SIZE, WHITE, FONT_THICKNESS)

        PREV_HAND_X = hand_x
        PREV_HAND_Y = hand_y

    elif pred_class == 0:
        PREV_HAND_Y = None
        PREV_HAND_X = None

# =========================================================


print(f"🎥 Sistem Navigasi YouTube Gestur V7 Aktif ({NUM_CLASSES} Kelas Final)...")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    label = ""
    
    if not results.multi_hand_landmarks:
        PREDICTION_HISTORY = []
        PREV_HAND_Y = None
        PREV_HAND_X = None
        
        text_nt = "❌ Tangan tidak terdeteksi"
        (text_width, text_height), baseline = cv2.getTextSize(text_nt, FONT_FACE, FONT_SIZE, FONT_THICKNESS + 2)
        bottom_right_x = w - text_width - 10
        bottom_right_y = h - 10
        
        cv2.putText(frame, text_nt, (bottom_right_x, bottom_right_y), FONT_FACE, FONT_SIZE, RED, FONT_THICKNESS + 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            input_data = normalize_keypoints(hand_landmarks)
            
            if input_data.ndim == 1: input_data = input_data[np.newaxis, ...]
            
            pred = model.predict(input_data, verbose=0)
            
            PREDICTION_HISTORY.append(pred)
            if len(PREDICTION_HISTORY) > HISTORY_SIZE:
                PREDICTION_HISTORY.pop(0)
            
            avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
            
            if avg_pred.ndim > 1 and avg_pred.shape[1] == NUM_CLASSES: 
                pred_class = np.argmax(avg_pred[0])
                confidence = avg_pred[0][pred_class]
            else:
                pred_class = 0 
                confidence = 0.0
            
            color = (0, 0, 255) 
            
            if confidence >= MIN_CONFIDENCE:
                
                if pred_class == 8:
                    handle_cursor_movement(frame, hand_landmarks, pred_class)
                    color = (255, 100, 255)
                
                elif pred_class != 0:
                    activate_gesture_action(pred_class)
                    color = (0, 255, 0)
                
                elif pred_class == 0:
                     color = (255, 255, 0)
                
                gesture_name_only = get_gesture_name(gestures[pred_class])
                label = f"AKTIF: {gesture_name_only} ({confidence*100:.1f}%)"
                
            else:
                color = (0, 165, 255)
                gesture_name_only = get_gesture_name(gestures.get(pred_class, 'Unknown'))
                label = f"Menganalisis: {gesture_name_only} ({confidence*100:.1f}%)"
            
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            if 'avg_pred' in locals() and avg_pred.ndim > 1:
                top_3_indices = np.argsort(avg_pred[0])[::-1][:3]
                
                cv2.putText(frame, "Top 3 Prediksi:", (w-150, 20), FONT_FACE, FONT_SIZE, WHITE, FONT_THICKNESS)
                
                for i, idx in enumerate(top_3_indices):
                    gesture_name_top = get_gesture_name(gestures.get(idx, 'Unknown'))
                    text_label = f"{gesture_name_top}: {avg_pred[0][idx]*100:.1f}%"
                    cv2.putText(frame, text_label, (w-150, 40 + i*20), FONT_FACE, FONT_SIZE, WHITE, FONT_THICKNESS)

    cv2.putText(frame, label, (20, 40), FONT_FACE, FONT_SIZE, WHITE, FONT_THICKNESS + 1)
    
    cv2.imshow('YouTube Gesture Controller V7 (22 Classes)', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()