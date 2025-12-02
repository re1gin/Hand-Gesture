-----

# 🚀 Proyek: YouTube Gesture Controller V6 (20 Gestur Final)

Sistem ini memungkinkan Anda mengontrol video YouTube, peramban Chrome, dan fungsi *mouse* **secara *real-time*** menggunakan **20 gerakan tangan** yang dideteksi oleh kamera, MediaPipe, dan Model Jaringan Saraf Tiruan (Neural Network) V6.

-----

## 1\. ⚙️ Prasyarat & Instalasi

Pastikan Anda telah menginstal **Python (3.8+)** di sistem Anda.

### A. Instalasi Pustaka

Buka Terminal atau Command Prompt, lalu jalankan perintah berikut untuk menginstal semua pustaka yang diperlukan:

```bash
pip install opencv-python mediapipe numpy tensorflow keras pyautogui webbrowser
```

### B. Struktur Folder Proyek V6

Pastikan struktur folder proyek Anda telah diperbarui untuk **versi 6 (20 Kelas)**:

```
YouTube_Gesture_Controller/
├── keypoint_collector_v6.py    # Pengumpul data (20 kelas)
├── keypoint_trainer_v6.py      # Pelatih model (20 kelas)
├── realtime_controller_v6.py   # Kontrol real-time
├── keypoint_dataset_v6/        # <-- Folder data baru (minimal 250 sampel per kelas)
└── keypoint_model_v6/          # <-- Model output: youtube_controller_mlp_v6.h5
```

-----

## 2\. 🖐️ Tahap Pelatihan Model (Training)

Model harus dilatih ulang untuk mengenali **20 gestur** baru (ID 0-19).

### A. Pengumpulan Data (Running `keypoint_collector_v6.py`)

1.  Jalankan *script* pengumpul data:
    ```bash
    python keypoint_collector_v6.py
    ```
2.  Ikuti instruksi di konsol. Anda perlu mengumpulkan data untuk **20 kelas gestur** baru (ID 0 hingga ID 19).
3.  **Target:** Kumpulkan $\ge 250$ sampel untuk setiap gestur. Pastikan variasi posisi tangan agar model **robust**.
4.  Tekan **[SPASI]** berulang kali untuk mengambil sampel.

### B. Pelatihan Model (Running `keypoint_trainer_v6.py`)

1.  Setelah semua data terkumpul, jalankan *script* pelatihan:
    ```bash
    python keypoint_trainer_v6.py
    ```
2.  Model **MLP** akan dilatih dengan **20 *output classes***.
3.  Model terbaik akan disimpan sebagai `keypoint_model_v6/youtube_controller_mlp_v6.h5`.

-----

## 3\. ✨ Cara Penggunaan (Real-Time Control)

Setelah model **V6** selesai dilatih, Anda siap menggunakannya.

1.  **Aktivasi Aplikasi:** Buka *browser* **Google Chrome** (atau aplikasi lain yang ingin Anda kontrol) dan pastikan jendela tersebut aktif (fokus).
2.  **Jalankan Kontroler:**
    ```bash
    python realtime_controller_v6_rapi.py
    ```
3.  Jendela kamera akan terbuka. Lakukan gestur yang Anda inginkan.
4.  Aksi akan dijalankan jika *confidence* $\ge 85\%$.

-----

## 4\. ✋ Daftar Gestur Kontrol V6 (20 Aksi Final)

Sistem ini memetakan **20 gestur tangan** (ID kelas 0 hingga 19) ke kontrol media, navigasi, dan fungsi *mouse* dinamis.

| ID Kelas | Gestur Tangan | Label Data | Aksi (Tombol/Perintah) | Tipe |
| :---: | :---: | :---: | :---: | :---: |
| 0 | **Telapak Terbuka** | `OPEN_PALM` | **Netral/Default** | Statis |
| 1 | **Tangan Mengepal** | `FIST` | `Spacebar` (Play/Pause) | Statis |
| 2 | **Jempol Ke Atas** | `THUMB_UP` | `Panah Kiri` ($\leftarrow$) (Rewind 5s) | Statis |
| 3 | **Jempol Ke Bawah** | `THUMB_DOWN` | `Panah Kanan` ($\rightarrow$) (Forward 5s) | Statis |
| 4 | **Telunjuk Ke Atas** | `INDEX_UP` | `Panah Atas` ($\uparrow$) (Volume Up) | Statis |
| 5 | **Telunjuk Ke Bawah** | `INDEX_DOWN` | `Panah Bawah` ($\downarrow$) (Volume Down) | Statis |
| 6 | **Tangan Berbentuk 'C'** | `C_SHAPE` | `C` (Subtitle On/Off) | Statis |
| 7 | **Tanda 'OK'** | `OK_SIGN` | `Click Kiri Mouse` | Statis |
| 8 | **Dua Jari Lurus** (Telunjuk & Tengah) | `TWO_FINGERS_UP` | **Arahkan Kursor** | **Dinamis (Mouse Move)** |
| 9 | **Tiga Jari Lurus** (Jempol & Kelingking terlipat) | `THREE_FINGERS_UP` | `M` (Mute/Unmute) | Statis |
| 10 | **Empat Jari Lurus** (Jempol terlipat) | `FOUR_FINGERS` | `F` (Fullscreen) | Statis |
| 11 | **Satu Jari Miring 90°** | `INDEX_SIDE_90` | `T` (Theater Mode) | Statis |
| 12 | **Dua Jari Miring 90°** | `TWO_FINGERS_SIDE_90` | `webbrowser.open()` (**Buka YouTube**) | Statis |
| 13 | **Dua Jari Miring Balik** | `TWO_FINGERS_SIDE_BACK` | `Enter` (Pilih/Kirim) | Statis |
| 14 | **Tiga Jari Miring 90°** | `THREE_FINGERS_SIDE_90` | `Ctrl + W` (Tutup Tab) | Statis |
| 15 | **Jari Kelingking Lurus** | `PINKY_UP` | `Esc` (Escape/Keluar) | Statis |
| 16 | **Bentuk 'L'** | `L_SHAPE` | `Shift + N` (Next Video) | Statis |
| 17 | **Bentuk 'Pistol'** | `GUN_SHAPE` | `Shift + P` (Previous Video) | Statis |
| 18 | **Scroll Atas (Statis)** | `SCROLL_UP` | `Scroll Mouse Up` (Cepat) | Statis |
| 19 | **Scroll Bawah (Statis)** | `SCROLL_DOWN` | `Scroll Mouse Down` (Cepat) | Statis |

***Catatan tentang Gestur Khusus:***

  * **Gestur Kursor (ID 8):** Gerakan kursor diatur secara **Dinamis** berdasarkan pergerakan pergelangan tangan Anda.
  * **Gestur Buka YouTube (ID 12):** Memiliki *cooldown* yang lebih lama (**10.0 detik**) untuk mencegah pembukaan *tab* berlebihan.

-----

## 🛑 Tips Pemecahan Masalah

  * **Aksi Tidak Bekerja:** Pastikan jendela aplikasi (misalnya, Chrome) yang ingin Anda kontrol sedang **aktif (fokus)**. `PyAutoGUI` mengirim *input* ke jendela yang sedang fokus.
  * **Kursor Bergerak Terlalu Cepat/Lambat:** Sesuaikan nilai **`CURSOR_SENSITIVITY`** di *script* *controller*.
  * **Aksi Statis Berulang:** Atur nilai **`ACTIVATION_COOLDOWN`** di *script* *controller* untuk mengontrol seberapa cepat aksi statis dapat diulang (default 0.5 detik).
