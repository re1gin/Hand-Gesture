
---

# 🎥 YouTube Gesture Controller V7 (22-Class Hand Gesture Navigation)

Sistem kontrol YouTube berbasis **gesture tangan** menggunakan:

* **MediaPipe Hands**
* **TensorFlow (MLP 22 kelas)**
* **OpenCV**
* **PyAutoGUI**
* **Custom smoothing + cursor acceleration**
* **Gesture cooldown management**

Versi ini mampu melakukan **navigasi penuh pada YouTube dan browser**, termasuk:

✔ Kontrol play/pause
✔ Volume
✔ Next / Previous
✔ Fullscreen / Theater
✔ Scroll
✔ Mute
✔ Back / Forward browser
✔ Close tab
✔ Buka YouTube otomatis
✔ Mode cursor (mouse) super smooth & wide

---

## ✨ Fitur Utama

### 🖐️ **22 Gestur Final (Kelas V7)**

Semua gesture telah dilatih dan terintegrasi:

| ID | Gesture               | Fungsi                   |
| -- | --------------------- | ------------------------ |
| 0  | OPEN_PALM             | Default / reset          |
| 1  | FIST                  | Play/Pause (Space)       |
| 2  | THUMB_UP              | Forward 10s              |
| 3  | THUMB_DOWN            | Rewind 10s               |
| 4  | INDEX_UP              | Volume Up                |
| 5  | INDEX_DOWN            | Volume Down              |
| 6  | C_SHAPE               | Subtitle toggle (C)      |
| 7  | OK_SIGN               | Mouse Click              |
| 8  | TWO_FINGERS_UP        | Cursor Mode (mouse)      |
| 9  | THREE_FINGERS_UP      | Mute (M)                 |
| 10 | FOUR_FINGERS          | Fullscreen (F)           |
| 11 | INDEX_SIDE_90         | Theater Mode (T)         |
| 12 | TWO_FINGERS_SIDE_90   | Open YouTube             |
| 13 | TWO_FINGERS_SIDE_BACK | Enter                    |
| 14 | THREE_FINGERS_SIDE_90 | Close Tab                |
| 15 | PINKY_UP              | Esc                      |
| 16 | L_SHAPE               | Next (Shift+N)           |
| 17 | GUN_SHAPE             | Previous (Shift+P)       |
| 18 | SCROLL_UP             | Scroll Up                |
| 19 | SCROLL_DOWN           | Scroll Down              |
| 20 | THUMB_LEFT            | Back Page (Alt+Left)     |
| 21 | THUMB_RIGHT           | Forward Page (Alt+Right) |

---

## 🖱️ **Mode Kursor Tingkat Lanjut (ID = 8)**

Kursor bergerak dengan:

* **Exponential smoothing (lebih halus dari linear smoothing)**
* **Soft acceleration**
* **Wide movement scaling**
* **Deadzone anti-jitter**
* **Auto-reset saat keluar dari mode cursor**

Hasilnya:

✔ Stabil
✔ Halus
✔ Tidak kaku
✔ Jangkauan luas
✔ Presisi tinggi

---

## 🧩 Struktur Teknis

### Algoritma Utama:

1. **MediaPipe Hand Tracking**
   Mendapatkan 21 landmark 3D tangan.

2. **Normalize Keypoints (63D)**

   * Translate ke pergelangan tangan
   * Normalize menggunakan jari tengah (landmark 9)

3. **MLP Gesture Classification (22 kelas)**
   Menggunakan model `.h5` custom.

4. **Prediction Smoothing**
   Mengambil rata-rata 5 frame (history buffer).

5. **Confidence Filtering**
   Gesture hanya dieksekusi jika confidence ≥ 0.85.

6. **Cooldown System**
   Untuk mencegah spam aksi.

7. **Gesture → Action Mapping**
   Menggunakan PyAutoGUI.

8. **Cursor System**
   Dengan smoothing + acceleration.

---

## 📦 Instalasi

### 1. Clone Repo

```bash
git clone https://github.com/your-repo/gesture-youtube-controller
cd gesture-youtube-controller
```

### 2. Install Dependencies

```bash
pip install opencv-python mediapipe tensorflow pyautogui numpy
```

### 3. Pastikan Model Tersedia

Model harus berada di:

```
keypoint_model_v7/youtube_controller_mlp_v7.h5
```

---

## ▶️ Cara Menjalankan

```bash
python main.py
```

Tekan **Q** untuk keluar.

---

## 🔧 Pengaturan Sensitivitas (Optional)

### Ubah sensitivitas kursor:

```python
CURSOR_SENSITIVITY = 3.5  # default
```

### Geser lebih cepat (misal layar besar):

```python
CURSOR_SENSITIVITY = 5.0
```

### Geser lebih stabil:

```python
CURSOR_SENSITIVITY = 2.8
```

---

## 🚀 Performance Tips

✔ Pastikan pencahayaan bagus
✔ Gunakan background yang kontras
✔ Gunakan webcam 30–60 FPS
✔ Jarak tangan 40–80cm dari kamera

---

## 📚 Roadmap (V8 – Optional)

Jika ditingkatkan lagi:

* Kalman Filter cursor tracking
* Adaptive sensitivity (AI)
* Drag-and-drop gesture
* Auto calibration per-user
* 30+ gesture recognition

---

## 👑 Penutup

YouTube Gesture Controller V7 adalah sistem navigasi berbasis gesture yang:

* Cepat
* Stabil
* Presisi
* Bebas latency
* Sangat ergonomis

Siap digunakan untuk:

* Kontrol YouTube
* Kontrol browser
* Presentasi
* Aplikasi hands-free

---
