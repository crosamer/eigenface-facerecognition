import kagglehub
import os

# Nama dataset di Kaggle Hub
KAGGLE_DATASET_REF = "hereisburak/pins-face-recognition"

print(f"Mulai mengunduh dataset '{KAGGLE_DATASET_REF}' dari Kaggle Hub...")

try:
    # Unduh versi terbaru dari dataset
    # path akan mengembalikan direktori tempat dataset diunduh dan diekstrak
    path = kagglehub.dataset_download(KAGGLE_DATASET_REF)

    print(f"Dataset berhasil diunduh ke: {path}")
    print("\nPetunjuk:")
    print("Dataset ini biasanya memiliki struktur folder di dalamnya.")
    print("Anda mungkin perlu menavigasi ke subfolder yang berisi gambar-gambar wajah yang sebenarnya.")
    print(f"Misalnya, cari folder seperti '{path}{os.sep}pins_face_recognition' atau serupa.")
    print("Folder ini akan Anda pilih sebagai 'Folder Dataset Training' di aplikasi GUI.")

except Exception as e:
    print(f"Terjadi kesalahan saat mengunduh dataset: {e}")
    print("Pastikan Anda sudah menginstal 'kagglehub' (pip install kagglehub) dan memiliki koneksi internet.")
