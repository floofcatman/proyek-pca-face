import os
import cv2
import pickle
import numpy as np
from sklearn.decomposition import PCA

IMAGE_SIZE = (100, 100)
PCA_COMPONENTS = 50

# Gunakan Haar Cascade lokal
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

X_list = []
y_list = []
dataset_dir = './dataset'

print("🔄 Memulai proses training lokal pada dataset...")

if not os.path.exists(dataset_dir):
    print("❌ Folder 'dataset' tidak ditemukan!")
    exit()

for label_name in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label_name)
    if not os.path.isdir(label_path):
        continue
        
    print(f"-> Memproses wajah: {label_name}")
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Jalankan deteksi wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            continue
            
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        cropped = img[y:y+h, x:x+w]
        
        # Preprocessing ke 100x100
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(cropped, IMAGE_SIZE)
        normalized = resized.astype('float32') / 255.0
        
        X_list.append(normalized.flatten())
        y_list.append(label_name)

if len(X_list) > 0:
    X_train = np.array(X_list)
    y_train = np.array(y_list)
    
    # Fit PCA
    n_comp = min(PCA_COMPONENTS, len(X_list))
    pca_model = PCA(n_components=n_comp)
    X_train_pca = pca_model.fit_transform(X_train)
    
    # Bungkus semua komponen model menjadi satu payload
    model_payload = {
        "pca_model": pca_model,
        "X_train_pca": X_train_pca,
        "y_train": y_train
    }
    
    # Simpan langsung ke dalam folder api/
    os.makedirs('./api', exist_ok=True)
    with open('./api/model_wajah.pkl', 'wb') as f:
        pickle.dump(model_payload, f)
        
    print("\n✅ SELESAI! Berkas 'api/model_wajah.pkl' berhasil dibuat.")
    print(f"Total wajah yang berhasil didaftarkan: {len(X_list)} sampel.")
else:
    print("❌ Gagal: Tidak ada wajah yang berhasil dideteksi dari dataset Anda.")