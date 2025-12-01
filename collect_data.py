import cv2
import os
import mediapipe as mp
import numpy as np
import time

# ================== Setup MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

gestures = {'rock': 'Batu', 'paper': 'Kertas', 'scissors': 'Gunting'}
DATA_DIR = 'dataset'
num_samples = 250
os.makedirs(DATA_DIR, exist_ok=True)
for g in gestures: os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

cap = cv2.VideoCapture(0)
time.sleep(1)

blank = np.zeros((128, 128, 3), np.uint8)
cv2.putText(blank, "Menunggu...", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

def show_preview(img):
    cv2.namedWindow('Preview Disimpan', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview Disimpan', 200, 200)
    cv2.imshow('Preview Disimpan', img)

os.system('cls' if os.name == 'nt' else 'clear')
print("PENGUMPUL DATA - BURST 10x + BACKGROUND HITAM".center(60))
available_gestures = list(gestures.keys())

while available_gestures:
    print("\nGesture belum selesai:")
    for i, g in enumerate(available_gestures, 1):
        n = len(os.listdir(os.path.join(DATA_DIR, g)))
        print(f"  {i}. {gestures[g]:<10} → {n}/250 {'SELESAI' if n>=250 else ''}")
    try:
        current = available_gestures[int(input(f"\nPilih (1-{len(available_gestures)}): ")) - 1]
    except: 
        print("Input salah!"); continue

    print(f"\n→ SEDANG: {gestures[current].upper()}")
    time.sleep(2)
    for i in range(3,0,-1):
        ret, f = cap.read()
        if ret:
            cv2.putText(f, str(i), (280,240), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,255), 15)
            cv2.imshow('Gesture Collector', f)
            cv2.waitKey(1000)

    count = len(os.listdir(os.path.join(DATA_DIR, current)))
    show_preview(blank)

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        preview = blank.copy()

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            x_min = y_min = 9999; x_max = y_max = 0
            for lm in results.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x*w), int(lm.y*h)
                x_min, x_max = min(x_min,x), max(x_max,x)
                y_min, y_max = min(y_min,y), max(y_max,y)

            pad = 60
            x1 = max(0, x_min-pad); y1 = max(0, y_min-pad)
            x2 = min(w, x_max+pad); y2 = min(h, y_max+pad)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                resized = cv2.resize(cropped, (100,100))
                black_bg = np.zeros((128,128,3), np.uint8)
                black_bg[14:114, 14:114] = resized
                preview = black_bg

            cv2.putText(frame, gestures[current], (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 3)
            cv2.putText(frame, f"{count}/250", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        else:
            cv2.putText(frame, "Tangan tidak terdeteksi", (70,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Bar perintah
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,420), (640,480), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "SPASI=Ambil 10 Foto  N=Lanjut  R=Ulangi  Q=Keluar", (10,455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        show_preview(preview)
        cv2.imshow('Gesture Collector', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            if results.multi_hand_landmarks:
                flash = frame.copy()
                cv2.putText(flash, "10", (260,260), cv2.FONT_HERSHEY_SIMPLEX, 7, (0,255,255), 18)
                cv2.imshow('Gesture Collector', flash); cv2.waitKey(400)
                print(f"\nBurst 10 foto {gestures[current]}...")
                for _ in range(10):
                    path = os.path.join(DATA_DIR, current, f"{count}.jpg")
                    cv2.imwrite(path, preview)
                    count += 1
                    print(f"   → {count}/250")
                    if os.name == 'nt':
                        import winsound
                        winsound.Beep(1400, 70)
                    time.sleep(0.22)
                print("Burst selesai!\n")
        elif k == ord('n'):
            if count >= num_samples and current in available_gestures:
                available_gestures.remove(current)
            break
        elif k == ord('r'):
            for f in os.listdir(os.path.join(DATA_DIR, current)):
                os.remove(os.path.join(DATA_DIR, current, f))
            count = 0
        elif k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit()

    if not available_gestures:
        print("\nSELESAI SEMUA GESTURE!")
        break

cap.release()
cv2.destroyAllWindows()