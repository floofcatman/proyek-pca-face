import os
import cv2
import pickle
import numpy as np
from sklearn.decomposition import PCA

# Pastikan ukuran ini sama dengan di index.py
IMAGE_SIZE = (100, 100) 

# Load pendeteksi wajah Haar Cascade
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

def preprocess_face(face_img):
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
    face_resized = cv2.resize(face_img, IMAGE_SIZE)
    face_eq = cv2.equalizeHist(face_resized)
    
    mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    center = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)
    
    face_masked = cv2.bitwise_and(face_eq, face_eq, mask=mask)
    return (face_masked.astype('float32') / 255.0).flatten()

def detect_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    
    # Ambil wajah yang ukurannya paling besar (wajah utama)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

def train_model():
    # Menggunakan prefix 'r' agar backslash pada Windows tidak dianggap sebagai escape character
    dataset_dir = r"C:\Users\Rizky\Downloads\proyek-pca-face\dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Folder '{dataset_dir}' tidak ditemukan!")
        return

    face_data = []
    labels = [] 
    print("Memulai ekstraksi fitur wajah dari dataset...")
    
    valid_images = 0
    
    # Menggunakan os.walk untuk menelusuri setiap sub-folder secara otomatis
    for root, dirs, files in os.walk(dataset_dir):
        # Mengambil nama sub-folder saat ini (misal: "Will Smith")
        person_name = os.path.basename(root)
        
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                cropped_face = detect_and_crop(img)
                
                if cropped_face is not None:
                    processed_vector = preprocess_face(cropped_face)
                    face_data.append(processed_vector)
                    labels.append(person_name)
                    valid_images += 1
                    print(f"Berhasil memproses: [{person_name}] -> {filename}")
                else:
                    print(f"Gagal deteksi wajah: [{person_name}] -> {filename}")

    if valid_images < 2:
        print("Dataset terlalu sedikit! Butuh minimal 2 wajah untuk melatih PCA.")
        return

    X = np.array(face_data)
    unique_people = len(set(labels))
    
    print(f"\nTotal dataset siap latih: {valid_images} gambar dari {unique_people} folder orang.")
    print("Melatih model PCA...")
    
    pca = PCA(n_components=0.95) 
    pca.fit(X)
    
    print(f"Model PCA berhasil dilatih! Jumlah komponen yang dipertahankan: {pca.n_components_}")

    model_data = {
        "pca_model": pca
    }
    
    output_filename = 'model_wajah.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nSelesai! Model disimpan sebagai '{output_filename}'.")

if __name__ == "__main__":
    train_model()