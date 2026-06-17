import streamlit as st
import cv2
import pickle
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="BioID - Face Verification", page_icon="🕵️", layout="centered")

IMAGE_SIZE = (100, 100)
SIMILARITY_THRESHOLD = 0.80


@st.cache_resource
def load_models():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    pca_model = None
    if os.path.exists('model_wajah.pkl'):
        with open('model_wajah.pkl', 'rb') as f:
            data = pickle.load(f)
            pca_model = data['pca_model']
            
    return face_cascade, pca_model

face_cascade, pca_model = load_models()

def detect_and_crop(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    return img_array[y:y+h, x:x+w]

def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    return (resized.astype('float32') / 255.0).flatten()


st.title("🕵️ BioID - Perbandingan Wajah")
st.markdown("Unggah foto masa kecil & foto masa sekarang untuk dibandingkan menggunakan algoritma Eigenfaces (PCA).")

if pca_model is None:
    st.error("❌ File 'model_wajah.pkl' tidak ditemukan di folder utama. Harap pastikan file tersebut ada.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Foto 1")
    file1 = st.file_uploader("Pilih Foto 1", type=['jpg', 'jpeg', 'png'], key="file1")
    if file1:
        img1_pil = Image.open(file1).convert('RGB')
        st.image(img1_pil, use_container_width=True)

with col2:
    st.subheader("Foto 2")
    file2 = st.file_uploader("Pilih Foto 2", type=['jpg', 'jpeg', 'png'], key="file2")
    if file2:
        img2_pil = Image.open(file2).convert('RGB')
        st.image(img2_pil, use_container_width=True)

if file1 and file2:
    if st.button("🚀 Bandingkan Wajah", use_container_width=True):
        with st.spinner("Mengekstrak fitur matriks..."):
            # Konversi gambar ke format numpy array untuk OpenCV
            img1_arr = np.array(img1_pil)
            img2_arr = np.array(img2_pil)
            
          
            face1 = detect_and_crop(img1_arr)
            face2 = detect_and_crop(img2_arr)
            
            if face1 is None:
                st.error("❌ Wajah tidak terdeteksi pada Foto 1.")
            elif face2 is None:
                st.error("❌ Wajah tidak terdeteksi pada Foto 2.")
            else:
                # Preprocessing
                flat1 = preprocess_face(face1)
                flat2 = preprocess_face(face2)
                
                # Proyeksi PCA
                vec1 = pca_model.transform([flat1])[0]
                vec2 = pca_model.transform([flat2])[0]
                
                # Kalkulasi Kemiripan
                cos_sim = float(cosine_similarity([vec1], [vec2])[0][0])
                eucl_dist = float(np.linalg.norm(vec1 - vec2))
                
                # Menampilkan Hasil
                st.divider()
                st.subheader("📊 Hasil Analisis")
                
                if cos_sim >= SIMILARITY_THRESHOLD:
                    st.success(f"**ORANG YANG SAMA!** Wajah teridentifikasi cocok.")
                else:
                    st.error(f"**BERBEDA ORANG.** Karakteristik wajah tidak cocok.")
                
                # Tampilkan Metrik
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Cosine Similarity", f"{cos_sim * 100:.1f}%")
                m_col2.metric("Jarak Euclidean", f"{eucl_dist:.2f}")
                m_col3.metric("Threshold (Batas)", f"{SIMILARITY_THRESHOLD * 100}%")
                
                st.progress(max(0.0, min(1.0, cos_sim)), text="Tingkat Kemiripan")