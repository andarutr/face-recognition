# Face Recognition System (Flask)

Aplikasi Web berbasis Flask untuk manajemen data wajah dan verifikasi wajah.

## Fitur
- **Database Wajah**: Penyimpanan dan pengelolaan data wajah.
- **Verifikasi Wajah**: Deteksi dan verifikasi wajah menggunakan webcam.
- **Registrasi**: Proses pendaftaran wajah baru.

## Instalasi dan Persiapan

### Prasyarat
- Python 3.6+
- pip
- Kamera Web (untuk fitur verifikasi)

### 1. Clone Repository
```bash
git clone <url-repository-anda>
cd face-recognition
```

### 2. Instalasi Dependensi
Buat environment virtual (opsional tapi direkomendasikan):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

Instal paket yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Database
Sistem ini menggunakan file JSON (`database.json`) untuk menyimpan data wajah. Pastikan file ini ada atau buatlah dengan struktur berikut (akan dibuat otomatis saat registrasi jika belum ada):
```json
{
    "W1": {
        "nama": "Nama Wajah",
        "wajah": "base64-encoded-image"
    }
}
```

## Cara Menjalankan Aplikasi

### Opsi 1: Mode Development (Recommended)
Menjalankan server Flask dengan auto-reload (terdeteksi perubahan file otomatis):
```bash
flask run
```

Akses aplikasi di: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Opsi 2: Menjalankan dengan Parameter Spesifik
```bash
set FLASK_APP=app.py
set FLASK_ENV=development
flask run --host=[IP_ADDRESS] --port=5000
```

## Struktur Project
```
face-recognition/
├── app.py              # Aplikasi utama Flask
├── requirements.txt    # Dependensi Python
├── database.json       # Database wajah (JSON)
├── static/             # File statis (CSS, JS, gambar)
│   └── css/
│       ├── style.css   # Style utama
│       └── bootstrap.min.css
└── templates/          # Template HTML
    ├── base.html       # Template dasar
    ├── index.html      # Halaman utama
    ├── register.html   # Halaman registrasi
    ├── verify.html     # Halaman verifikasi
```

## API Endpoints
- `GET /register`: Tampilkan formulir registrasi.
- `POST /register`: Daftarkan wajah baru.
- `GET /verify`: Tampilkan halaman verifikasi.
- `POST /verify`: Lakukan verifikasi wajah dari gambar yang diunggah.
- `GET /verify_webcam`: Lakukan verifikasi menggunakan webcam.

## Debugging dan Logs
Flask secara otomatis akan menampilkan error di terminal saat mode `development` aktif. Pastikan terminal tetap terbuka saat menjalankan aplikasi untuk melihat output dan error.
