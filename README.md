
-----

# 🚀 Proyek: YouTube Gesture Controller V4 (16 Gestur)

Sistem ini memungkinkan Anda mengontrol video YouTube, peramban Chrome, dan fungsi *mouse* **secara *real-time*** menggunakan **16 gerakan tangan** yang dideteksi oleh kamera, MediaPipe, dan Model Jaringan Saraf Tiruan (Neural Network) V4.

-----

## 1\. ⚙️ Prasyarat & Instalasi

Pastikan Anda telah menginstal **Python (3.8+)** di sistem Anda.

### A. Instalasi Pustaka

Buka Terminal atau Command Prompt, lalu jalankan perintah berikut untuk menginstal semua pustaka yang diperlukan:

```bash
pip install opencv-python mediapipe numpy tensorflow keras pyautogui webbrowser
```

### B. Struktur Folder Proyek V4

Pastikan struktur folder proyek Anda telah diperbarui untuk versi 4:

```
YouTube_Gesture_Controller/
├── keypoint_collector_v4.py    # Pengumpul data (16 kelas)
├── keypoint_trainer_v4.py      # Pelatih model (16 kelas)
├── realtime_controller_v4.py   # Kontrol real-time
├── keypoint_dataset_v4/        # <-- Folder data baru (minimal 250 sampel per kelas)
└── keypoint_model_v4/          # <-- Model output: youtube_controller_mlp.h5
```

-----

## 2\. 🖐️ Tahap Pelatihan Model (Training)

Model harus dilatih ulang untuk mengenali 16 gestur yang baru.

### A. Pengumpulan Data (Running `keypoint_collector_v4.py`)

1.  Jalankan *script* pengumpul data:
    ```bash
    python keypoint_collector_v4.py
    ```
2.  Ikuti instruksi di konsol. Anda perlu mengumpulkan data untuk **16 kelas gestur** baru.
3.  **Target:** Kumpulkan $\ge 250$ sampel untuk setiap gestur. Pastikan variasi posisi tangan agar model **robust** (kuat).
4.  Tekan **[SPASI]** berulang kali untuk mengambil sampel.

### B. Pelatihan Model (Running `keypoint_trainer_v4.py`)

1.  Setelah semua data terkumpul, jalankan *script* pelatihan:
    ```bash
    python keypoint_trainer_v4.py
    ```
2.  Model **MLP (Multi-Layer Perceptron)** akan dilatih dengan **16 *output classes***.
3.  Model terbaik akan disimpan ke dalam folder `keypoint_model_v4/`.

-----

## 3\. ✨ Cara Penggunaan (Real-Time Control)

Setelah model **V4** selesai dilatih dan disimpan, Anda siap menggunakannya.

1.  **Aktivasi Chrome:** Buka *browser* **Google Chrome** (atau aplikasi lain yang ingin Anda kontrol) dan pastikan jendela tersebut aktif (fokus).
2.  **Jalankan Kontroler:**
    ```bash
    python realtime_controller_v4.py
    ```
3.  Jendela kamera akan terbuka, menampilkan tangan Anda dan status deteksi.
4.  Lakukan gestur. Aksi akan dijalankan jika *confidence* deteksi gestur statis $\ge 85\%$.

-----

## 4\. ✋ Daftar Gestur Kontrol V4 (16 Aksi)

Sistem ini memetakan 16 gestur tangan ke kontrol media, navigasi peramban, dan fungsi *mouse* dinamis.

| ID Kelas | Gestur Tangan | Label Data | Aksi (Tombol/Perintah) | Tipe |
| :---: | :---: | :---: | :---: | :---: |
| 0 | **Telapak Tangan Terbuka** | `open_palm` | **Netral** | Statis |
| 1 | **Tangan Mengepal** | `fist` | `Spacebar` (Play/Pause) | Statis |
| 2 | **Jempol Ke Atas** | `thumb_up` | `Panah Kiri` ($\leftarrow$) (Rewind) | Statis |
| 3 | **Jempol Ke Bawah** | `thumb_down` | `Panah Kanan` ($\rightarrow$) (Forward) | Statis |
| 4 | **Telunjuk Ke Atas** | `index_up` | `Panah Atas` ($\uparrow$) (Volume Up) | Statis |
| 5 | **Telunjuk Ke Bawah** | `index_down` | `Panah Bawah` ($\downarrow$) (Volume Down) | Statis |
| 6 | **Tangan Berbentuk 'C'** | `c_shape` | `M` (Mute/Unmute) | Statis |
| 7 | **Tanda 'OK'** | `ok_sign` | `Enter` (Pilih/Klik) | Statis |
| 8 | **Dua Jari Lurus** | `two_fingers_up` | **Arahkan Kursor** | **Dinamis (Mouse Move)** |
| 9 | **Tiga Jari Lurus** | `three_fingers_up` | **Scroll** | **Dinamis (Mouse Scroll)** |
| 10 | **Empat Jari Lurus** | `four_fingers` | `F` (Fullscreen) | Statis |
| 11 | **Satu Jari Miring 90°** | `index_side_90` | `Tab` (Navigasi Next Element) | Statis |
| 12 | **Dua Jari Miring 90°** | `two_fingers_side_90` | `webbrowser.open()` (Buka YouTube Baru) | Statis |
| 13 | **Tiga Jari Miring 90°** | `three_fingers_side_90` | `Ctrl + W` (Tutup Tab/YouTube) | Statis |
| 14 | **Jari Kelingking Lurus** | `pinky_up` | `Esc` (Escape/Keluar) | Statis |
| 15 | ***Swipe* Atas/Bawah** | `swipe_up_down` | **Scroll Cepat** | **Dinamis (Mouse Scroll)** |

***Catatan tentang Gestur Dinamis (ID 8, 9, 15):***
Gestur dinamis tidak menggunakan gerakan tangan yang di-traktiran (dipelajari) oleh model, melainkan **model mendeteksi bentuk tangan statisnya**, lalu *script* `realtime_controller_v4.py` menggunakan **posisi pergelangan tangan** saat gestur tersebut aktif untuk menghitung gerakan kursor (`pyautogui.move()`) atau gulir (`pyautogui.scroll()`).

-----

## 🛑 Tips Pemecahan Masalah

  * **Kursor Bergerak Terlalu Cepat/Lambat:** Sesuaikan nilai **`CURSOR_SENSITIVITY`** di *script* `realtime_controller_v4.py`.
  * **Scroll Tidak Akurat:** Sesuaikan nilai **`SCROLL_SENSITIVITY`** dan **`scroll_amount`** di *script* `realtime_controller_v4.py`.
  * **Aksi Tidak Bekerja:** Pastikan jendela **Google Chrome** aktif dan berada di depan. `PyAutoGUI` mengirim *input* ke jendela yang sedang fokus.
