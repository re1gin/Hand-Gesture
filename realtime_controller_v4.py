import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import pyautogui
import webbrowser
import os
import threading

# ======================================================================
# 1. PENGATURAN SISTEM & GLOBAL CONFIG
# ======================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.FAILSAFE = False 

# ======================================================================
# 2. SETUP UI PARAMETERS
# ======================================================================
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
FONT_THICKNESS = 1
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
CURSOR_COLOR = (255, 0, 255) 
# ======================================================================

# ======================================================================
# 3. SETUP MEDIAPIPE HANDS
# ======================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
# ======================================================================

# ======================================================================
# 4. CLASS MULTI-THREADING KAMERA
# ======================================================================
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
# ======================================================================

# ======================================================================
# 5. FUNGSI NORMALISASI KEYPOINT
# ======================================================================
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
# ======================================================================

# ======================================================================
# 6. PEMUATAN MODEL & SETUP GESTUR (V7 - 22 Kelas)
# ======================================================================
MODEL_PATH = 'keypoint_model_v7/youtube_controller_mlp_v7.h5'
NUM_CLASSES = 22

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, 63), dtype=np.float32)
    model(dummy_input, training=False) 
    print("✅ Model dimuat dan GPU Siap.")
except Exception as e:
    print(f"Error: Model '{MODEL_PATH}' tidak ditemukan atau gagal dimuat. {e}")
    exit()

gestures = {
    0:'OPEN_PALM (Default)', 1:'FIST (Play/Pause)', 2:'THUMB_UP (Forward)', 
    3:'THUMB_DOWN (Rewind)', 4:'INDEX_UP (Vol Up)', 5:'INDEX_DOWN (Vol Down)',
    6:'C_SHAPE (Subtitle)', 7:'OK_SIGN (Click)', 
    8:'TWO_FINGERS_UP (Cursor)', 
    9:'THREE_FINGERS_UP (Mute)', 10:'FOUR_FINGERS (Fullscreen)', 
    11:'INDEX_SIDE_90 (Teater Mode)', 12:'TWO_FINGERS_SIDE_90 (Open YT)', 
    13:'TWO_FINGERS_SIDE_BACK (Enter)', 14:'THREE_FINGERS_SIDE_90 (Close Tab)',
    15:'PINKY_UP (Esc)',
    16:'L_SHAPE (Next)', 17:'GUN_SHAPE (Previous)', 
    18:'SCROLL_UP (Static)', 19:'SCROLL_DOWN (Static)',
    20:'THUMB_LEFT (Back Page)', 21:'THUMB_RIGHT (Forward Page)',
}

def get_gesture_name(gesture_string):
    return gesture_string.split(' (')[0]
# ======================================================================

# ======================================================================
# 7. VARIABEL KONTROL & COOLDOWN
# ======================================================================
vs = WebcamVideoStream(src=0).start()
time.sleep(1.0) 

PREDICTION_HISTORY = []
HISTORY_SIZE = 5 
MIN_CONFIDENCE = 0.85 
LAST_ACTIVATION = time.time()
ACTIVATION_COOLDOWN = 1.0 
LAST_OPEN_YT = time.time()
OPEN_YT_COOLDOWN = 10.0 

# Variabel Kursor
PREV_HAND_Y = None
PREV_HAND_X = None
CURSOR_SENSITIVITY = 4.0 
DEADZONE = 6             

# VARIABEL SMOOTHING DITAMBAHKAN KEMBALI
SMOOTHING_FACTOR = 0.5  
GLOBAL_SMOOTH_DX = 0     
GLOBAL_SMOOTH_DY = 0     

SCROLL_AMOUNT = 300

prev_frame_time = 0
new_frame_time = 0
# ======================================================================

# ======================================================================
# 8. FUNGSI AKSI STATIS (PyAutoGUI)
# ======================================================================
def activate_gesture_action(gesture_id):
    global LAST_ACTIVATION, LAST_OPEN_YT
    
    if time.time() - LAST_ACTIVATION < ACTIVATION_COOLDOWN and gesture_id not in [8, 12]:
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
            print(f"✅ AKSI AKTIF: {gestures[gesture_id]}") 
            LAST_OPEN_YT = time.time() 
        return
    
    action = action_map.get(gesture_id)
    if action:
        action()
        print(f"✅ AKSI AKTIF: {gestures[gesture_id]}") 
        LAST_ACTIVATION = time.time() 
# ======================================================================

# ======================================================================
# 9. FUNGSI AKSI DINAMIS (KURSOR ID 8) - SMOOTHING DAN DEADZONE
# ======================================================================
def handle_cursor_movement(frame_shape, hand_landmarks):
    global PREV_HAND_Y, PREV_HAND_X
    global GLOBAL_SMOOTH_DX, GLOBAL_SMOOTH_DY # Gunakan variabel smoothing global
    global CURSOR_SENSITIVITY, DEADZONE, SMOOTHING_FACTOR

    h, w = frame_shape
    wrist_lm = hand_landmarks.landmark[0]

    hand_y = wrist_lm.y * h
    hand_x = wrist_lm.x * w

    if PREV_HAND_X is None:
        PREV_HAND_X = hand_x
        PREV_HAND_Y = hand_y
        # Reset Smoothing saat kursor baru diaktifkan
        GLOBAL_SMOOTH_DX = 0
        GLOBAL_SMOOTH_DY = 0
        return

    # 1. Hitung Perubahan Mentah (Raw Delta)
    dx_raw = hand_x - PREV_HAND_X
    dy_raw = hand_y - PREV_HAND_Y

    # 2. Deadzone (Mengabaikan Getaran Kecil)
    if abs(dx_raw) < DEADZONE: dx_raw = 0
    if abs(dy_raw) < DEADZONE: dy_raw = 0

    # 3. Exponential Smoothing (Stabilisasi Utama)
    # Ini adalah bagian yang ditambahkan kembali
    GLOBAL_SMOOTH_DX = GLOBAL_SMOOTH_DX * (1 - SMOOTHING_FACTOR) + dx_raw * SMOOTHING_FACTOR
    GLOBAL_SMOOTH_DY = GLOBAL_SMOOTH_DY * (1 - SMOOTHING_FACTOR) + dy_raw * SMOOTHING_FACTOR
    
    # 4. Soft Acceleration (Sensitivitas Dinamis)
    speed = np.sqrt(dx_raw**2 + dy_raw**2)
    accel = 1 + min(speed * 0.015, 1.0) 

    # 5. Scaling Akhir (Menggunakan smoothed delta)
    dx = GLOBAL_SMOOTH_DX * CURSOR_SENSITIVITY * accel
    dy = GLOBAL_SMOOTH_DY * CURSOR_SENSITIVITY * accel

    # Gerakkan kursor (dx tidak dikalikan -1 agar tidak mirror)
    pyautogui.move(int(dx), int(dy), _pause=False)

    # 6. Update Posisi Sebelumnya
    PREV_HAND_X = hand_x
    PREV_HAND_Y = hand_y
# ======================================================================

print(f"🚀 Sistem Aktif: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')} | Multi-threaded | {NUM_CLASSES} Kelas")

# ======================================================================
# 10. MAIN LOOP
# ======================================================================
while True:
    frame = vs.read()
    if frame is None: break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    label = ""
    color = (0, 165, 255)

    if not results.multi_hand_landmarks:
        # Reset State jika Tangan Hilang
        PREDICTION_HISTORY = []
        PREV_HAND_Y = None 
        PREV_HAND_X = None
        # Reset Smoothing State
        GLOBAL_SMOOTH_DX = 0
        GLOBAL_SMOOTH_DY = 0
        cv2.putText(frame, "Tangan tidak terdeteksi", (w-200, h-10), FONT_FACE, FONT_SIZE, RED, 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            input_data = normalize_keypoints(hand_landmarks)
            if input_data.ndim == 1: input_data = input_data[np.newaxis, ...]
            
            pred = model(input_data, training=False).numpy()
            
            PREDICTION_HISTORY.append(pred)
            if len(PREDICTION_HISTORY) > HISTORY_SIZE:
                PREDICTION_HISTORY.pop(0)
            avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
            
            pred_class = np.argmax(avg_pred[0])
            confidence = avg_pred[0][pred_class]
            
            if confidence >= MIN_CONFIDENCE:
                
                if pred_class == 8: # CURSOR MODE (Dinamis, Smoothing & Deadzone ON)
                    handle_cursor_movement((h, w), hand_landmarks)
                    label = f"CURSOR ACTIVE ({confidence*100:.1f}%)"
                    color = CURSOR_COLOR 
                    
                else: 
                    # Reset kursor dan smoothing state saat beralih ke aksi statis/netral
                    PREV_HAND_Y = None 
                    PREV_HAND_X = None
                    GLOBAL_SMOOTH_DX = 0
                    GLOBAL_SMOOTH_DY = 0
                    
                    if pred_class == 0:
                        label = f"NETRAL ({confidence*100:.1f}%)"
                        color = WHITE
                    else:
                        activate_gesture_action(pred_class)
                        label = f"AKSI: {get_gesture_name(gestures[pred_class])} ({confidence*100:.1f}%)"
                        color = GREEN
                        
            else:
                label = f"Menganalisis... ({get_gesture_name(gestures[pred_class])}: {confidence*100:.1f}%)"
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2))

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), FONT_FACE, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, label, (10, 50), FONT_FACE, 0.7, color, 2)
    
    cv2.imshow('Optimized Gesture Controller (Smooth Cursor)', frame)

    if cv2.waitKey(1) == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()