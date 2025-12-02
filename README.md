
-----

# 🚀 Proyek: YouTube Gesture Controller V4 (16 Gestur)

Sistem ini memungkinkan Anda mengontrol video YouTube, peramban Chrome, dan fungsi *mouse* **secara *real-time*** menggunakan **16 gerakan tangan** yang dideteksi oleh kamera, MediaPipe, dan Model Jaringan Saraf Tiruan (Neural Network) V4.

-----

## 1\. ⚙️ Prasyarat & Instalasi

Pastikan Anda memiliki Python (disarankan versi 3.8+) terinstal di sistem Anda.

### A. Instalasi Pustaka

Buka Terminal atau Command Prompt, lalu jalankan perintah berikut untuk menginstal semua pustaka yang diperlukan:

```bash
pip install opencv-python mediapipe numpy tensorflow keras pyautogui webbrowser
```

### B. Struktur Folder Proyek V4

Pastikan struktur folder proyek Anda telah diperbarui dan terlihat seperti ini:

```
YouTube_Gesture_Controller/
├── keypoint_collector_v4.py
├── keypoint_trainer_v4.py
├── realtime_controller_v4.py
├── keypoint_dataset_v4/  <-- Folder data baru (16 kelas)
└── keypoint_model_v4/    <-- Folder model baru
```

-----

## 2\. 🖐️ Tahap Pelatihan Model (Training)

Karena Anda menggunakan 16 gestur baru, model harus dilatih ulang.

### A. Pengumpulan Data (Running `keypoint_collector_v4.py`)

1.  Jalankan *script* pengumpul data:
    ```bash
    python keypoint_collector_v4.py
    ```
2.  Ikuti instruksi di konsol. Anda perlu mengumpulkan data untuk **16 kelas gestur** baru.
3.  Pilih gestur, lalu posisikan tangan Anda di depan kamera.
4.  Tekan tombol **[SPASI]** berulang kali (target **250 kali per gestur**) untuk mengambil sampel *keypoint*. Pastikan posisi tangan Anda bervariasi agar model menjadi **robust**.
5.  Ulangi proses ini hingga semua **16 gestur** mencapai target 250 sampel di folder `keypoint_dataset_v4`.

### B. Pelatihan Model (Running `keypoint_trainer_v4.py`)

1.  Setelah semua data terkumpul, jalankan *script* pelatihan:
    ```bash
    python keypoint_trainer_v4.py
    ```
2.  Model **MLP (Multi-Layer Perceptron)** akan dikompilasi dengan **16 *output classes***.
3.  Model terbaik akan dilatih, dan yang paling akurat akan disimpan sebagai **`keypoint_model_v4/youtube_controller_mlp.h5`**.

-----

## 3\. ✨ Cara Penggunaan (Real-Time Control)

Setelah model **V4** selesai dilatih, Anda siap menggunakannya.

1.  **Pastikan Chrome Aktif:** Buka *browser* **Google Chrome** (atau aplikasi lain yang ingin Anda kontrol).
2.  **Jalankan Kontroler:**
    ```bash
    python realtime_controller_v4.py
    ```
3.  Jendela kamera akan terbuka.
4.  Lakukan gestur yang Anda inginkan.
5.  Jika *confidence* gestur statis $\ge 85\%$, aksi yang dipetakan akan dijalankan. Untuk gestur dinamis (*Cursor* dan *Scroll*), aksi akan dijalankan terus menerus selama gestur terdeteksi.

-----

## 4\. ✋ Daftar Gestur Kontrol V4 (16 Aksi)

Sistem ini memetakan 16 gestur tangan ke kontrol media, navigasi peramban, dan fungsi *mouse* dinamis.

| ID Kelas | Gestur Tangan | Label Data | Aksi (Tombol/Perintah) | Tipe |
| :---: | :---: | :---: | :---: | :---: |
| 0 | **Telapak Tangan Terbuka** | `open_palm` | **Netral** | Statis |
| 1 | **Tangan Mengepal** | `fist` | `Spacebar` | Statis (Play/Pause) |
| 2 | **Jempol Ke Atas** | `thumb_up` | `Panah Kiri` ($\leftarrow$) | Statis (Rewind) |
| 3 | **Jempol Ke Bawah** | `thumb_down` | `Panah Kanan` ($\rightarrow$) | Statis (Forward) |
| 4 | **Telunjuk Ke Atas** | `index_up` | `Panah Atas` ($\uparrow$) | Statis (Volume Up) |
| 5 | **Telunjuk Ke Bawah** | `index_down` | `Panah Bawah` ($\downarrow$) | Statis (Volume Down) |
| 6 | **Tangan Berbentuk 'C'** | `c_shape` | `M` | Statis (Mute/Unmute) |
| 7 | **Tanda 'OK'** | `ok_sign` | `Enter` | Statis (Pilih/Klik) |
| 8 | **Dua Jari Lurus** (Telunjuk & Tengah) | `two_fingers_up` | **Arahkan Kursor** | **Dinamis (Mouse Move)** |
| 9 | **Tiga Jari Lurus** (Jempol & Kelingking terlipat) | `three_fingers_up` | **Scroll** | **Dinamis (Mouse Scroll)** |
| 10 | **Empat Jari Lurus** (Jempol terlipat) | `four_fingers` | `F` | Statis (Fullscreen) |
| 11 | **Satu Jari Miring 90°** | `index_side_90` | `Tab` | Statis (Navigasi Next Element) |
| 12 | **Dua Jari Miring 90°** | `two_fingers_side_90` | `webbrowser.open()` | Statis (Buka YouTube Baru) |
| 13 | **Tiga Jari Miring 90°** | `three_fingers_side_90` | `Ctrl + W` | Statis (Tutup Tab/YouTube) |
| 14 | **Jari Kelingking Lurus** | `pinky_up` | `Esc` | Statis (Escape/Keluar) |
| 15 | ***Swipe* Atas/Bawah** | `swipe_up_down` | **Scroll** | **Dinamis (Mouse Scroll)** |

***Catatan tentang Gestur Dinamis (ID 8, 9, 15):***

  * Model **hanya mendeteksi bentuk tangan statis** untuk ID 8, 9, dan 15.
  * *Script* **`realtime_controller_v4.py`** kemudian **menggunakan posisi pergelangan tangan** saat gestur tersebut aktif untuk menghitung gerakan kursor (`pyautogui.move()`) atau gulir (`pyautogui.scroll()`).

-----

## 🛑 Tips Pemecahan Masalah

  * **Kursor Bergerak Terlalu Cepat/Lambat:** Sesuaikan nilai **`CURSOR_SENSITIVITY`** di *script* `realtime_controller_v4.py`.
  * **Scroll Tidak Akurat:** Sesuaikan nilai **`SCROLL_SENSITIVITY`** dan **`scroll_amount`** di *script* `realtime_controller_v4.py`.
  * **Aksi Tidak Bekerja:** Pastikan jendela **Google Chrome** aktif dan berada di depan. `PyAutoGUI` mengirim *input* ke jendela yang sedang fokus.