import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../templates')

# Sediakan Haar Cascade untuk gambar uji yang diunggah pengguna
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

IMAGE_SIZE = (100, 100)
SIMILARITY_THRESHOLD = 0.80

# Load pre-trained model secara instan dari berkas pickle
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_wajah.pkl')

pca_model = None
X_train_pca = None
y_train = None
is_model_ready = False

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            data_model = pickle.load(f)
        pca_model = data_model["pca_model"]
        X_train_pca = data_model["X_train_pca"]
        y_train = data_model["y_train"]
        is_model_ready = True
        print("✅ Berhasil memuat pre-trained model PCA ke memori.")
    except Exception as e:
        print(f"❌ Gagal membaca berkas model: {str(e)}")
else:
    print("❌ Berkas 'model_wajah.pkl' tidak ditemukan di folder api/.")

def preprocess_face(face_img):
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_img, IMAGE_SIZE)
    return (face_resized.astype('float32') / 255.0).flatten()

def detect_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not is_model_ready:
        return jsonify({"error": "Model pintar belum siap di server. Pastikan berkas model_wajah.pkl sudah ikut di-deploy."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File gambar tidak valid."}), 400
        
    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal membaca format gambar."}), 400
            
        cropped_face = detect_and_crop(img)
        if cropped_face is None:
            return jsonify({"error": "Wajah tidak terdeteksi, silakan coba foto lain."}), 400
            
        flat_face = preprocess_face(cropped_face)
        
        # Proyeksi wajah input ke ruang komponen PCA yang sudah dilatih
        face_pca = pca_model.transform([flat_face])
        
        # Hitung skor kedekatan fitur memakai Cosine Similarity
        similarities = cosine_similarity(face_pca, X_train_pca)[0]
        best_match_idx = np.argmax(similarities)
        highest_score = similarities[best_match_idx]
        
        if highest_score >= SIMILARITY_THRESHOLD:
            predicted_name = y_train[best_match_idx]
            return jsonify({
                "success": True,
                "match": predicted_name,
                "score": float(highest_score),
                "message": f"Wajah dikenali sebagai {predicted_name}"
            })
        else:
            return jsonify({
                "success": True,
                "match": "Tidak dikenal",
                "score": float(highest_score),
                "message": "Wajah tidak dikenali (Skor kemiripan di bawah standar)"
            })
            
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500