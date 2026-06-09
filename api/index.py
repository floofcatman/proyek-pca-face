import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../templates')

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Jalur API tidak ditemukan. Cek vercel.json Anda."}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Terjadi kesalahan internal di server backend Vercel."}), 500

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

IMAGE_SIZE = (100, 100)
SIMILARITY_THRESHOLD = 0.80

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_wajah.pkl')

pca_model = None
is_model_ready = False

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            data_model = pickle.load(f)
        pca_model = data_model["pca_model"]
        is_model_ready = True
    except Exception as e:
        print(f"Error memuat model: {str(e)}")

def preprocess_face(face_img):
    # 1. Konversi ke Grayscale jika belum
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
    # 2. Resize ke ukuran seragam
    face_resized = cv2.resize(face_img, IMAGE_SIZE)
    
    # 3. Histogram Equalization: Mengatasi masalah pencahayaan yang berbeda
    face_eq = cv2.equalizeHist(face_resized)
    
    # 4. Masking Oval/Lingkaran: Menghapus noise dari background
    mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    center = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)
    
    # Terapkan mask (area di luar lingkaran akan menjadi hitam pekat / 0)
    face_masked = cv2.bitwise_and(face_eq, face_eq, mask=mask)
    
    # 5. Normalisasi dan Flatten menjadi array 1D
    return (face_masked.astype('float32') / 255.0).flatten()

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

@app.route('/compare', methods=['POST'])
@app.route('/api/compare', methods=['POST'])
@app.route('/api/index.py', methods=['POST'])
@app.route('/<path:any_path>', methods=['POST'])
def compare_faces(any_path=None):
    if not is_model_ready:
        return jsonify({"error": "Model PCA belum siap. File model_wajah.pkl tidak terbaca."}), 500

    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Dua foto dibutuhkan untuk proses perbandingan."}), 400
        
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    def process_upload(file_obj):
        if file_obj.filename == '':
            return None, "Ada berkas yang kosong."
        img_bytes = file_obj.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Gagal membaca format gambar."
        cropped = detect_and_crop(img)
        if cropped is None:
            return None, "Wajah tidak terdeteksi di salah satu foto."
        return preprocess_face(cropped), None

    flat1, err1 = process_upload(file1)
    if err1: return jsonify({"error": err1}), 400
        
    flat2, err2 = process_upload(file2)
    if err2: return jsonify({"error": err2}), 400
        
    try:
        vec1 = pca_model.transform([flat1])[0]
        vec2 = pca_model.transform([flat2])[0]
        
        cos_sim = float(cosine_similarity([vec1], [vec2])[0][0])
        eucl_dist = float(np.linalg.norm(vec1 - vec2))
        
        is_match = bool(cos_sim >= SIMILARITY_THRESHOLD)
        
        if cos_sim >= 0.88: confidence = "Tinggi"
        elif cos_sim >= 0.75: confidence = "Sedang"
        else: confidence = "Rendah"
            
        return jsonify({
            "success": True,
            "is_match": is_match,
            "similarity": cos_sim,
            "distance": round(eucl_dist, 3),
            "threshold": SIMILARITY_THRESHOLD,
            "confidence": confidence
        })
        
    except Exception as e:
        return jsonify({"error": f"Kesalahan komputasi: {str(e)}"}), 500