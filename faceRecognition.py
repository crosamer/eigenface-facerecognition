# --- Library Bawaan Python ---
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time

# --- WAJIB INSTALL LIBRARY INI DULU ---
# pip install numpy
# pip install Pillow
import numpy as np
from PIL import Image, ImageTk

# ====================================================================================
# BAGIAN YANG HARUS KAMU IMPLEMENTASIKAN DAN PERBAIKI
# ====================================================================================

# Ganti variabel ini menjadi True untuk menggunakan implementasi manualmu saat pengumpulan tugas.
# Untuk sekarang, biarkan False agar kita bisa memastikan seluruh bagian aplikasi lain berjalan benar.
GUNAKAN_IMPLEMENTASI_MANUAL = True

def hitung_eigen_manual(covariance_matrix):
    """
    [IMPLEMENTASI MANUAL DARI KAMU]
    Fungsi ini adalah implementasi manualmu menggunakan Power Iteration.
    Kemungkinan besar ada masalah stabilitas numerik di sini yang menyebabkan
    hasilnya tidak akurat. Kamu bisa fokus memperbaiki fungsi ini secara terpisah.
    """
    print(">>> (MANUAL) Menghitung Nilai Eigen & Vektor Eigen...")
    A = covariance_matrix.copy()
    n = A.shape[0]
    eigenvalues_list = []
    eigenvectors_list = []
    max_eigen_to_find = n
    max_iter = 1000
    tolerance = 1e-6

    for k in range(max_eigen_to_find):
        # Inisialisasi vektor acak
        v = np.random.rand(n)
        
        # Normalisasi vektor v
        norm_v = np.sqrt(np.sum(v**2))
        if norm_v == 0: break
        v = v / norm_v
        
        lambda_prev = 0.0
        
        for _ in range(max_iter):
            # Lakukan perkalian matriks-vektor: Av = A @ v
            Av = A @ v
            
            # Hitung norm
            norm_Av = np.sqrt(np.sum(Av**2))
            if norm_Av < 1e-9: break
            
            # Normalisasi untuk mendapatkan vektor eigen berikutnya
            v_next = Av / norm_Av
            
            # Estimasi eigenvalue
            lambda_val = v_next.T @ Av
            
            # Cek konvergensi
            if np.sqrt(np.sum((v_next - v)**2)) < tolerance and abs(lambda_val - lambda_prev) < tolerance:
                break
            
            v = v_next
            lambda_prev = lambda_val
        
        if norm_Av > 1e-9:
            eigenvalues_list.append(lambda_val)
            eigenvectors_list.append(v)
            # Deflasi matriks
            A = A - lambda_val * np.outer(v, v)
        else:
            break
            
    eigenvalues = np.array(eigenvalues_list)
    eigenvectors = np.array(eigenvectors_list).T if eigenvectors_list else np.array([])
    
    # ... (Proses sorting manualmu bisa ditambahkan di sini jika perlu) ...
    return eigenvalues, eigenvectors

def hitung_eigen(covariance_matrix):
    """
    Fungsi "saklar" untuk memilih antara implementasi manual atau library.
    Ini membantumu memvalidasi seluruh alur kerja aplikasi tanpa terganggu
    oleh kemungkinan bug di implementasi manual.
    """
    if GUNAKAN_IMPLEMENTASI_MANUAL:
        # Panggil fungsi manualmu untuk pengujian akhir
        return hitung_eigen_manual(covariance_matrix)
    else:
        # Gunakan fungsi library yang stabil untuk pengembangan & pengujian
        print(">>> (LIBRARY) Menghitung Nilai Eigen & Vektor Eigen menggunakan np.linalg.eig...")
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Urutkan dari yang terbesar untuk konsistensi
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

def hitung_jarak_euclidean(vec1, vec2):
    """Menghitung jarak euclidean antara dua vektor."""
    # Implementasi manualmu sudah benar.
    jarak_kuadrat = 0.0
    for i in range(len(vec1)):
        diff = vec1[i] - vec2[i]
        jarak_kuadrat += diff * diff
    return np.sqrt(jarak_kuadrat)

# ====================================================================================
# BAGIAN LOGIKA UTAMA (BACKEND)
# ====================================================================================

class FaceRecognitionEngine:
    def __init__(self, size=(100, 100)):
        self.size = size
        self.is_trained = False
        # Variabel-variabel ini akan disimpan dan dimuat dari file model
        self.psi_mean_face = None
        self.eigenfaces = None
        self.weights_training = None
        self.nama_file_training = None
        self.original_dataset_path = None

    def train(self, folder_path):
        """
        [FIXED] Melakukan training pada dataset gambar di folder, termasuk semua subfolder.
        """
        self.original_dataset_path = folder_path
        
        images = []
        self.nama_file_training = []
        
        print(f"Memulai scanning gambar di: {folder_path}...")
        # Gunakan os.walk untuk menelusuri semua folder di dalam path yang diberikan
        for subdir, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Path lengkap ke file gambar
                    full_path = os.path.join(subdir, file)
                    
                    # Buat path relatif dari folder_path utama.
                    # Ini penting agar saat disimpan dan dimuat, lokasinya tetap benar.
                    # Contoh: 'pins_Cristiano Ronaldo\gambar.jpg'
                    relative_path = os.path.relpath(full_path, folder_path)
                    
                    try:
                        img = Image.open(full_path).convert('L').resize(self.size)
                        images.append(np.array(img).flatten())
                        self.nama_file_training.append(relative_path)
                    except Exception as e:
                        print(f"Gagal memproses {full_path}: {e}")
        
        if not images: raise ValueError("Tidak ada gambar yang valid ditemukan di dalam folder dataset atau subfoldernya.")
        print(f"Ditemukan {len(images)} gambar untuk training.")
        
        matriks_training = np.array(images)
        M, N = matriks_training.shape

        # 2. Hitung rataan wajah (mean face)
        self.psi_mean_face = np.mean(matriks_training, axis=0)

        # 3. Normalisasi setiap wajah dengan mengurangi rataan
        phi_normalized_faces = matriks_training - self.psi_mean_face

        # 4. Hitung matriks kovarian (menggunakan trik efisiensi)
        L = np.dot(phi_normalized_faces, phi_normalized_faces.T)

        # 5. Hitung nilai eigen dan vektor eigen (menggunakan fungsi "saklar")
        eigenvalues, eigenvectors = hitung_eigen(L)
        if len(eigenvalues) == 0: raise ValueError("Gagal menghitung eigenvalue. Coba periksa dataset.")

        # 6. Hitung Eigenface dari vektor eigen
        self.eigenfaces = np.dot(phi_normalized_faces.T, eigenvectors).T
        
        # 7. Hitung bobot fitur untuk setiap gambar training
        self.weights_training = np.dot(self.eigenfaces, phi_normalized_faces.T)
        
        self.is_trained = True
        print("Training selesai.")

    def recognize(self, file_path_test):
        """Mengenali gambar uji berdasarkan model yang sudah di-training."""
        if not self.is_trained: raise ValueError("Model belum di-training atau dimuat.")
            
        img_test = Image.open(file_path_test).convert('L').resize(self.size)
        vec_test = np.array(img_test).flatten()
        phi_normalized_test = vec_test - self.psi_mean_face

        # Proyeksikan gambar uji ke ruang eigenface untuk mendapatkan bobotnya
        weight_test = np.dot(self.eigenfaces, phi_normalized_test)

        # Cari jarak terdekat
        jarak_terkecil = float('inf')
        indeks_terdekat = -1
        # Iterasi melalui setiap kolom dari matriks bobot training
        for i in range(self.weights_training.shape[1]):
            weight_train = self.weights_training[:, i]
            jarak = hitung_jarak_euclidean(weight_test, weight_train)
            if jarak < jarak_terkecil:
                jarak_terkecil = jarak
                indeks_terdekat = i
        
        # Tentukan nilai batas (threshold) kemiripan. Nilai ini bisa disesuaikan.
        THRESHOLD_KEMIRIPAN = 3500 
        
        if jarak_terkecil < THRESHOLD_KEMIRIPAN:
            nama_file_hasil = self.nama_file_training[indeks_terdekat]
            pesan_hasil = f"Paling Mirip: {os.path.basename(nama_file_hasil)}\nJarak: {jarak_terkecil:.2f}"
            return nama_file_hasil, pesan_hasil
        else:
            return None, f"Tidak ditemukan wajah yang cocok.\nJarak terdekat: {jarak_terkecil:.2f}"

    def save_model(self, path):
        """Menyimpan model yang sudah di-training ke file .npz."""
        if not self.is_trained: raise ValueError("Tidak ada model untuk disimpan.")
        np.savez(path,
                 psi_mean_face=self.psi_mean_face,
                 eigenfaces=self.eigenfaces,
                 weights_training=self.weights_training,
                 nama_file_training=self.nama_file_training,
                 size=self.size,
                 original_dataset_path=self.original_dataset_path)
        print(f"Model berhasil disimpan di {path}")

    def load_model(self, path):
        """Memuat model dari file .npz."""
        data = np.load(path, allow_pickle=True)
        self.psi_mean_face = data['psi_mean_face']
        self.eigenfaces = data['eigenfaces']
        self.weights_training = data['weights_training']
        self.nama_file_training = data['nama_file_training']
        self.size = tuple(data['size'])
        self.original_dataset_path = str(data['original_dataset_path']) 
        self.is_trained = True
        print(f"Model berhasil dimuat dari {path}")
        print(f"Path dataset asli: {self.original_dataset_path}")

# ====================================================================================
# BAGIAN ANTARMUKA (GUI) DENGAN PERBAIKAN
# ====================================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition using Eigenface")
        self.geometry("850x600")

        self.engine = FaceRecognitionEngine()
        self.folder_dataset = ""
        self.file_test = ""
        
        # --- Layout ---
        # Kolom Kiri: Pengaturan
        left_frame = tk.Frame(self, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="PENGATURAN", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Frame untuk Model Training
        model_frame = tk.LabelFrame(left_frame, text="1. Model Training")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.btn_dataset = tk.Button(model_frame, text="Pilih Folder Dataset", command=self.pilih_dataset)
        self.btn_dataset.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_dataset = tk.Label(model_frame, text="Folder belum dipilih", wraplength=220)
        self.lbl_dataset.pack(fill=tk.X, padx=5)

        self.btn_train = tk.Button(model_frame, text="Train Model", command=self.jalankan_training_thread, bg="#FFA500")
        self.btn_train.pack(fill=tk.X, padx=5, pady=5)
        
        model_ops_frame = tk.Frame(model_frame)
        model_ops_frame.pack(fill=tk.X)
        self.btn_save_model = tk.Button(model_ops_frame, text="Simpan", command=self.simpan_model)
        self.btn_save_model.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        self.btn_load_model = tk.Button(model_ops_frame, text="Muat", command=self.muat_model)
        self.btn_load_model.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        
        # Frame untuk Uji & Hasil
        tk.Label(left_frame, text="UJI & HASIL", font=("Helvetica", 14, "bold")).pack(pady=10)
        self.btn_image = tk.Button(left_frame, text="2. Pilih Gambar Uji", command=self.pilih_gambar_uji)
        self.btn_image.pack(fill=tk.X, pady=5)
        
        self.btn_recognize = tk.Button(left_frame, text="3. JALANKAN PENGENALAN", command=self.jalankan_pengenalan, height=2, bg="green", fg="white")
        self.btn_recognize.pack(fill=tk.X, pady=10)
        
        self.lbl_status = tk.Label(left_frame, text="Status: Siap", font=("Helvetica", 10, "italic"), fg="gray")
        self.lbl_status.pack(fill=tk.X, pady=5)
        self.lbl_result = tk.Label(left_frame, text="Hasil akan muncul di sini...", font=("Helvetica", 12), wraplength=220, justify=tk.LEFT)
        self.lbl_result.pack(fill=tk.X, pady=5)

        # Kolom Kanan: Tampilan Gambar
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        test_img_frame = tk.LabelFrame(right_frame, text="Test Image", font=("Helvetica", 12))
        test_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.panel_test = tk.Label(test_img_frame)
        self.panel_test.pack(expand=True)
        result_img_frame = tk.LabelFrame(right_frame, text="Closest Result", font=("Helvetica", 12))
        result_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.panel_result = tk.Label(result_img_frame)
        self.panel_result.pack(expand=True)

    def set_ui_state(self, state):
        """Mengatur state tombol (enabled/disabled) untuk mencegah aksi saat proses berjalan."""
        for btn in [self.btn_dataset, self.btn_train, self.btn_save_model, self.btn_load_model, self.btn_image, self.btn_recognize]:
            btn.config(state=state)

    def pilih_dataset(self):
        self.folder_dataset = filedialog.askdirectory(title="Pilih Folder Dataset")
        if self.folder_dataset:
            self.lbl_dataset.config(text=f"Folder: ...{os.path.basename(self.folder_dataset)}")
            self.lbl_status.config(text="Status: Dataset dipilih. Siap untuk training.", fg="blue")

    def jalankan_training_thread(self):
        """Memulai proses training di thread terpisah agar GUI tidak 'Not Responding'."""
        if not self.folder_dataset:
            messagebox.showwarning("Peringatan", "Pilih folder dataset terlebih dahulu!")
            return
        
        self.set_ui_state(tk.DISABLED)
        self.lbl_status.config(text="Status: Training... Mohon tunggu.", fg="orange")
        thread = threading.Thread(target=self._thread_train)
        thread.start()

    def _thread_train(self):
        """Fungsi yang dijalankan oleh thread untuk proses training."""
        try:
            start_time = time.time()
            self.engine.train(self.folder_dataset)
            end_time = time.time()
            
            # Kembali ke main thread untuk update UI setelah selesai
            self.after(0, self._on_training_complete, end_time - start_time)
        except Exception as e:
            self.after(0, self._on_training_error, e)

    def _on_training_complete(self, duration):
        """Callback saat training sukses."""
        self.set_ui_state(tk.NORMAL)
        self.lbl_status.config(text=f"Status: Training selesai ({duration:.2f} d).", fg="green")
        messagebox.showinfo("Sukses", "Training model berhasil diselesaikan.")

    def _on_training_error(self, error):
        """Callback saat training gagal."""
        self.set_ui_state(tk.NORMAL)
        self.lbl_status.config(text="Status: Gagal training!", fg="red")
        messagebox.showerror("Error Training", str(error))

    def simpan_model(self):
        if not self.engine.is_trained:
            messagebox.showerror("Error", "Model belum di-training. Tidak ada yang bisa disimpan.")
            return
        try:
            path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ Model", "*.npz")], title="Simpan Model")
            if path:
                self.engine.save_model(path)
                messagebox.showinfo("Sukses", f"Model berhasil disimpan di {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan model: {e}")

    def muat_model(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("NPZ Model", "*.npz")], title="Muat Model")
            if path:
                self.engine.load_model(path)
                self.folder_dataset = self.engine.original_dataset_path
                self.lbl_status.config(text="Status: Model dimuat. Siap mengenali.", fg="green")
                self.lbl_dataset.config(text=f"Model untuk: ...{os.path.basename(self.folder_dataset)}")
                messagebox.showinfo("Sukses", f"Model berhasil dimuat dari {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model: {e}")
    
    def pilih_gambar_uji(self):
        self.file_test = filedialog.askopenfilename(title="Pilih Gambar Uji", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if self.file_test:
            self.tampilkan_gambar(self.file_test, self.panel_test)
            self.lbl_result.config(text="Gambar uji dipilih. Siap untuk pengenalan.")

    def jalankan_pengenalan(self):
        if not self.engine.is_trained:
            messagebox.showwarning("Peringatan", "Model belum di-training atau dimuat."); return
        if not self.file_test:
            messagebox.showwarning("Peringatan", "Pilih gambar uji terlebih dahulu."); return
        
        try:
            nama_file_hasil, pesan_hasil = self.engine.recognize(self.file_test)
            self.lbl_result.config(text=pesan_hasil)
            if nama_file_hasil:
                path_hasil = os.path.join(self.folder_dataset, nama_file_hasil)
                self.tampilkan_gambar(path_hasil, self.panel_result)
            else:
                self.panel_result.config(image='')
                self.panel_result.image = None
        except Exception as e:
            messagebox.showerror("Error Pengenalan", str(e))

    def tampilkan_gambar(self, path, panel):
        """Fungsi untuk menampilkan gambar di panel GUI."""
        try:
            img = Image.open(path)
            img.thumbnail((350, 350)) # Resize gambar agar pas di panel
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk
        except FileNotFoundError:
            messagebox.showerror("Error", f"File gambar tidak ditemukan di: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Tidak bisa menampilkan gambar: {e}")

# ====================================================================================
# JALANKAN APLIKASI
# ====================================================================================

if __name__ == '__main__':
    app = App()
    app.mainloop()