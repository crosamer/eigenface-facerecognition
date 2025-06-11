# --- Library Bawaan Python ---
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time

# --- WAJIB INSTALL LIBRARY INI DULU ---
# pip install numpy
# pip install Pillow
# pip install opencv-python
import numpy as np
from PIL import Image, ImageTk, ImageOps
import cv2

# Ganti variabel ini menjadi True untuk menggunakan implementasi manualmu saat pengumpulan tugas.
GUNAKAN_IMPLEMENTASI_MANUAL = True

def hitung_eigen_manual(covariance_matrix):
    """Implementasi manual Power Iteration milikmu."""
    print(">>> (MANUAL) Menghitung Nilai Eigen & Vektor Eigen...")
    A = covariance_matrix.copy()
    n = A.shape[0]
    eigenvalues_list = []
    eigenvectors_list = []
    for k in range(n):
        v = np.random.rand(n)
        norm_v = np.sqrt(np.sum(v**2)); v = v / norm_v if norm_v > 0 else v
        lambda_prev = 0.0
        for _ in range(1000):
            Av = A @ v
            norm_Av = np.sqrt(np.sum(Av**2))
            if norm_Av < 1e-9: break
            v_next = Av / norm_Av
            lambda_val = v_next.T @ Av
            if np.sqrt(np.sum((v_next - v)**2)) < 1e-6 and abs(lambda_val - lambda_prev) < 1e-6: break
            v = v_next; lambda_prev = lambda_val
        if norm_Av > 1e-9:
            eigenvalues_list.append(lambda_val); eigenvectors_list.append(v)
            A = A - lambda_val * np.outer(v, v)
        else: break
    eigenvalues = np.array(eigenvalues_list)
    eigenvectors = np.array(eigenvectors_list).T if eigenvectors_list else np.array([])
    return eigenvalues, eigenvectors

def hitung_eigen(covariance_matrix):
    """Fungsi "saklar" untuk memilih metode perhitungan eigen."""
    if GUNAKAN_IMPLEMENTASI_MANUAL:
        return hitung_eigen_manual(covariance_matrix)
    else:
        print(">>> (LIBRARY) Menghitung Nilai Eigen & Vektor Eigen menggunakan np.linalg.eig...")
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

def hitung_jarak_euclidean(vec1, vec2):
    """Menghitung jarak euclidean antara dua vektor."""
    jarak_kuadrat = np.sum((vec1 - vec2)**2)
    return np.sqrt(jarak_kuadrat)

class FaceRecognitionEngine:
    def __init__(self, size=(100, 100)):
        self.size = size
        self.is_trained = False
        self.psi_mean_face = None
        self.eigenfaces = None
        self.weights_training = None
        self.labels = None
        self.relative_paths = None
        self.original_dataset_path = None
        # Muat classifier untuk deteksi wajah. Pastikan file XML ada di folder yang sama.
        try:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        except Exception as e:
            messagebox.showerror("Error Kritis", "Tidak dapat memuat 'haarcascade_frontalface_default.xml'. Pastikan file ada di folder yang sama dengan aplikasi.")
            raise e

    def _preprocess_image(self, img_path):
        """Fungsi pra-pemrosesan dengan deteksi wajah dan cropping."""
        img_cv = cv2.imread(img_path)
        if img_cv is None: return None
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # [FIX] Membuat parameter deteksi lebih toleran
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, # Diubah dari 4 ke 3 agar lebih banyak wajah terdeteksi
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # print(f"Peringatan: Tidak ada wajah yang terdeteksi di {os.path.basename(img_path)}. Melewati gambar ini.")
            return None
            
        (x, y, w, h) = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        
        img_pil = Image.fromarray(face_crop)
        img_pil = ImageOps.equalize(img_pil)
        img_pil = img_pil.resize(self.size)
        return np.array(img_pil).flatten()

    def train(self, folder_path):
        """Melatih model pada SEMUA wajah yang terdeteksi di dalam subfolder dataset."""
        self.original_dataset_path = folder_path
        images, self.labels, self.relative_paths = [], [], []
        skipped_count = 0
        processed_count = 0
        print(f"Memulai scanning & deteksi wajah di: {folder_path}...")
        
        # Hitung total file gambar untuk progress bar
        total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
        
        for subdir, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(subdir, file)
                    label = os.path.basename(subdir).replace("pins_", "").replace("_", " ")
                    try:
                        processed_face = self._preprocess_image(full_path)
                        if processed_face is not None:
                            images.append(processed_face)
                            self.labels.append(label)
                            self.relative_paths.append(os.path.relpath(full_path, folder_path))
                            processed_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        skipped_count += 1
                        print(f"Gagal memproses {full_path}: {e}")
        
        print(f"Proses scanning selesai. {processed_count} wajah berhasil diproses, {skipped_count} gambar dilewati.")
        if not images: raise ValueError("Tidak ada wajah yang berhasil dideteksi dan diproses di dalam folder dataset.")
        print(f"Melatih model dengan {len(images)} wajah dari {len(set(self.labels))} artis...")
        
        matriks_training = np.array(images)
        self.psi_mean_face = np.mean(matriks_training, axis=0)
        phi_normalized_faces = matriks_training - self.psi_mean_face
        L = np.dot(phi_normalized_faces, phi_normalized_faces.T)
        eigenvalues, eigenvectors = hitung_eigen(L)
        if len(eigenvalues) == 0: raise ValueError("Gagal menghitung eigenvalue.")

        self.eigenfaces = np.dot(phi_normalized_faces.T, eigenvectors).T
        
        for i in range(self.eigenfaces.shape[0]):
            norm = np.linalg.norm(self.eigenfaces[i])
            if norm > 0: self.eigenfaces[i] = self.eigenfaces[i] / norm

        self.weights_training = np.dot(self.eigenfaces, phi_normalized_faces.T)
        self.is_trained = True
        print("Training selesai.")

    def recognize(self, file_path_test):
        if not self.is_trained: raise ValueError("Model belum di-training atau dimuat.")
        
        vec_test = self._preprocess_image(file_path_test)
        if vec_test is None:
            raise ValueError("Wajah tidak dapat dideteksi pada gambar uji.")
        
        phi_normalized_test = vec_test - self.psi_mean_face
        weight_test = np.dot(self.eigenfaces, phi_normalized_test)

        jarak_terkecil = float('inf')
        indeks_terdekat = -1
        for i in range(self.weights_training.shape[1]):
            weight_train = self.weights_training[:, i]
            jarak = hitung_jarak_euclidean(weight_test, weight_train)
            if jarak < jarak_terkecil:
                jarak_terkecil = jarak
                indeks_terdekat = i
        
        label_hasil = self.labels[indeks_terdekat]
        path_hasil = self.relative_paths[indeks_terdekat]
        return label_hasil, jarak_terkecil, path_hasil

    def save_model(self, path):
        if not self.is_trained: raise ValueError("Tidak ada model untuk disimpan.")
        np.savez(path, psi_mean_face=self.psi_mean_face, eigenfaces=self.eigenfaces,
                weights_training=self.weights_training, labels=self.labels,
                relative_paths=self.relative_paths, size=self.size,
                original_dataset_path=self.original_dataset_path)
        print(f"Model berhasil disimpan di {path}")

    def load_model(self, path):
        data = np.load(path, allow_pickle=True)
        self.psi_mean_face = data['psi_mean_face']
        self.eigenfaces = data['eigenfaces']
        self.weights_training = data['weights_training']
        self.labels = data['labels']
        self.relative_paths = data['relative_paths']
        self.size = tuple(data['size'])
        if 'original_dataset_path' in data: self.original_dataset_path = str(data['original_dataset_path'])
        else: self.original_dataset_path = None
        self.is_trained = True
        print(f"Model berhasil dimuat dari {path}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Pengenalan Wajah - Eigenface")
        self.geometry("900x650")

        try:
            self.engine = FaceRecognitionEngine()
        except Exception as e:
            self.destroy() 
            return

        self.file_test = ""
        self.dataset_path = ""
        self.last_recognition_result = None

        main_frame = tk.Frame(self)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(main_frame, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        self.left_frame.pack_propagate(False)
        
        model_frame = ttk.LabelFrame(self.left_frame, text="1. Latih atau Muat Model")
        model_frame.pack(fill=tk.X, pady=5)
        self.btn_dataset = ttk.Button(model_frame, text="Pilih Folder Dataset Utama", command=self.pilih_dataset)
        self.btn_dataset.pack(fill=tk.X, padx=10, pady=5)
        self.btn_train = ttk.Button(model_frame, text="Latih Model", command=self.jalankan_training_thread)
        self.btn_train.pack(fill=tk.X, padx=10, pady=5)
        model_ops_frame = tk.Frame(model_frame)
        model_ops_frame.pack(fill=tk.X, pady=(0, 5))
        self.btn_save_model = ttk.Button(model_ops_frame, text="Simpan Model", command=self.simpan_model)
        self.btn_save_model.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.btn_load_model = ttk.Button(model_ops_frame, text="Muat Model", command=self.muat_model)
        self.btn_load_model.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        recog_frame = ttk.LabelFrame(self.left_frame, text="2. Lakukan Pengenalan")
        recog_frame.pack(fill=tk.X, pady=10)
        self.btn_image = ttk.Button(recog_frame, text="Pilih Gambar untuk Dikenali", command=self.pilih_gambar_uji)
        self.btn_image.pack(fill=tk.X, padx=10, pady=5)
        self.btn_recognize = ttk.Button(recog_frame, text="Kenali Wajah Ini", command=self.jalankan_pengenalan)
        self.btn_recognize.pack(fill=tk.X, padx=10, pady=5)

        slider_frame = ttk.LabelFrame(self.left_frame, text="3. Atur Tingkat Kemiripan (Threshold)")
        slider_frame.pack(fill=tk.X, pady=5)
        self.threshold_var = tk.DoubleVar(value=5000)
        self.slider = ttk.Scale(slider_frame, from_=0, to=10000, orient=tk.HORIZONTAL, variable=self.threshold_var, command=self.update_slider_label)
        self.slider.pack(fill=tk.X, padx=10, pady=5)
        self.lbl_slider = ttk.Label(slider_frame, text=f"Jarak Maksimal: {self.threshold_var.get():.2f}")
        self.lbl_slider.pack()

        result_frame = ttk.LabelFrame(self.left_frame, text="Hasil Identifikasi")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.lbl_result_identity = ttk.Label(result_frame, text="??", font=("Helvetica", 24, "bold"), anchor="center")
        self.lbl_result_identity.pack(pady=10)
        self.lbl_result_details = ttk.Label(result_frame, text="Pilih gambar untuk memulai", anchor="center")
        self.lbl_result_details.pack(pady=5)
        self.lbl_status = ttk.Label(self, text="Status: Siap", relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        test_img_frame = ttk.LabelFrame(right_frame, text="Gambar Uji")
        test_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.panel_test = ttk.Label(test_img_frame, anchor="center")
        self.panel_test.pack(expand=True)
        result_img_frame = ttk.LabelFrame(right_frame, text="Gambar Paling Mirip dari Dataset")
        result_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.panel_result = ttk.Label(result_img_frame, anchor="center")
        self.panel_result.pack(expand=True)
    
    def set_ui_state(self, state):
        """Mengatur state semua tombol dan slider dengan benar."""
        widgets_to_toggle = [
            self.btn_dataset, self.btn_train, self.btn_save_model, 
            self.btn_load_model, self.btn_image, self.btn_recognize, self.slider
        ]
        for widget in widgets_to_toggle:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass 

    def update_slider_label(self, value):
        self.lbl_slider.config(text=f"Jarak Maksimal: {float(value):.2f}")
        if self.last_recognition_result:
            self.evaluasi_hasil(self.last_recognition_result)

    def pilih_dataset(self):
        path = filedialog.askdirectory(title="Pilih Folder Dataset Utama")
        if path:
            self.dataset_path = path
            self.lbl_status.config(text=f"Dataset dipilih: ...{os.path.basename(path)}. Siap untuk dilatih.")

    def jalankan_training_thread(self):
        if not self.dataset_path:
            messagebox.showwarning("Peringatan", "Pilih folder dataset utama terlebih dahulu!"); return
        self.set_ui_state(tk.DISABLED)
        self.lbl_status.config(text="Status: Training... Ini mungkin butuh waktu lama.")
        threading.Thread(target=self._thread_train, daemon=True).start()

    def _thread_train(self):
        try:
            start_time = time.time()
            self.engine.train(self.dataset_path)
            duration = time.time() - start_time
            self.after(0, self._on_training_complete, duration)
        except Exception as e:
            self.after(0, self._on_training_error, e)

    def _on_training_complete(self, duration):
        self.set_ui_state(tk.NORMAL)
        self.lbl_status.config(text=f"Status: Training selesai ({duration:.2f} detik). Model siap digunakan/disimpan.")
        messagebox.showinfo("Sukses", f"Training model berhasil.")

    def _on_training_error(self, error):
        self.set_ui_state(tk.NORMAL)
        self.lbl_status.config(text="Status: Gagal training!")
        messagebox.showerror("Error Training", str(error))

    def simpan_model(self):
        if not self.engine.is_trained:
            messagebox.showerror("Error", "Model belum dilatih. Tidak ada yang bisa disimpan."); return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("Eigenface Model", "*.npz")], title="Simpan Model")
        if path: self.engine.save_model(path); messagebox.showinfo("Sukses", "Model berhasil disimpan.")

    def muat_model(self):
        path = filedialog.askopenfilename(filetypes=[("Eigenface Model", "*.npz")], title="Muat Model")
        if path:
            try:
                self.engine.load_model(path)
                if self.engine.original_dataset_path:
                    self.dataset_path = self.engine.original_dataset_path
                self.lbl_status.config(text=f"Status: Model dimuat untuk {len(set(self.engine.labels))} artis. Siap mengenali.")
                messagebox.showinfo("Sukses", "Model berhasil dimuat.")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat model: {e}")
    
    def pilih_gambar_uji(self):
        path = filedialog.askopenfilename(title="Pilih Gambar Uji", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.file_test = path
            self.tampilkan_gambar(self.file_test, self.panel_test)
            self.lbl_result_identity.config(text="??", foreground="black")
            self.lbl_result_details.config(text="Klik 'Kenali Wajah Ini' untuk memulai.")
            self.panel_result.config(image=''); self.panel_result.image = None

    def jalankan_pengenalan(self):
        if not self.engine.is_trained: 
            messagebox.showwarning("Peringatan", "Latih atau muat model terlebih dahulu.")
            return
        if not self.file_test: 
            messagebox.showwarning("Peringatan", "Pilih gambar uji terlebih dahulu.")
            return
        
        try:
            self.last_recognition_result = self.engine.recognize(self.file_test)
            self.evaluasi_hasil(self.last_recognition_result)
        except Exception as e:
            messagebox.showerror("Error Pengenalan", str(e))

    def evaluasi_hasil(self, result):
        label, jarak, path_hasil_relatif = result
        threshold = self.threshold_var.get()

        # Clear result image first
        self.panel_result.config(image='')
        self.panel_result.image = None

        if jarak < threshold:
            self.lbl_result_identity.config(text=label, foreground="green")
            self.lbl_result_details.config(text=f"Jarak kemiripan: {jarak:.2f} (< {threshold:.2f})")
            # Only show matched image if below threshold
            if self.dataset_path:
                path_hasil_absolut = os.path.join(self.dataset_path, path_hasil_relatif)
                self.tampilkan_gambar(path_hasil_absolut, self.panel_result)
        else:
            self.lbl_result_identity.config(text="Tidak Dikenali", foreground="red")
            self.lbl_result_details.config(text=f"Jarak kemiripan: {jarak:.2f} (≥ {threshold:.2f})")

    def tampilkan_gambar(self, path, panel):
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk
        except Exception as e:
            print(f"Error menampilkan gambar {path}: {e}")

if __name__ == '__main__':
    app = App()
    app.mainloop()