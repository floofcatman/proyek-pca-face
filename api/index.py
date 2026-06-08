import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi Flask, pointing ke folder templates di luar folder api/
app = Flask(__name__, template_folder='../templates')

# Memuat Haar Cascade bawaan dari instalasi OpenCV
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Konstanta Algoritma sesuai persyaratan
IMAGE_SIZE = (100, 100) # Akan menghasilkan vektor 10.000 dimensi
PCA_COMPONENTS = 50
SIMILARITY_THRESHOLD = 0.80

# State Global untuk menyimpan model (Di-cache di memori instance Vercel)
pca_model = None
X_train_pca = None
y_train = []
is_model_trained = False

def preprocess_face(face_img):
    """
    Pra-pemrosesan:
    1. Konversi ke Grayscale.
    2. Resize ke 100x100.
    3. Normalisasi piksel ke rentang 0-1.
    4. Ratakan (flatten) menjadi vektor 10.000 dimensi.
    """
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    face_resized = cv2.resize(face_img, IMAGE_SIZE)
    face_normalized = face_resized.astype('float32') / 255.0
    face_flattened = face_normalized.flatten()
    return face_flattened

def detect_and_crop(img):
    """Mendeteksi wajah menggunakan Haar Cascade dan memotongnya."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
        
    # Jika terdapat lebih dari 1 wajah, ambil wajah dengan ukuran area terbesar
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face

def initialize_and_train_system():
    """
    Dipanggil saat cold-start serverless function. 
    Akan memindai folder 'dataset/', melatih model PCA dalam memori.
    """
    global pca_model, X_train_pca, y_train, is_model_trained
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, '../dataset')
    
    X_list = []
    y_list = []
    
    if not os.path.exists(dataset_dir):
        print("Peringatan: Folder dataset tidak ditemukan.")
        return

    # Membaca data gambar dari folder (nama folder = nama orang/label)
    for label_name in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_path):
            continue
            
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            cropped = detect_and_crop(img)
            if cropped is not None:
                flat_face = preprocess_face(cropped)
                X_list.append(flat_face)
                y_list.append(label_name)
                
    if len(X_list) > 0:
        X_train = np.array(X_list)
        y_train = np.array(y_list)
        
        # Fit PCA: gunakan batas minimum antara 50 komponen atau jumlah total gambar yang ada
        n_comp = min(PCA_COMPONENTS, len(X_list))
        pca_model = PCA(n_components=n_comp)
        X_train_pca = pca_model.fit_transform(X_train)
        is_model_trained = True
        print(f"Sistem berhasil dilatih dengan {len(X_list)} sampel wajah.")
    else:
        print("Peringatan: Tidak ada sampel wajah valid di dalam dataset.")

# Menjalankan fungsi pelatihan saat inisialisasi aplikasi (Cold-start Vercel)
initialize_and_train_system()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not is_model_trained:
        return jsonify({
            "error": "Sistem belum siap. Dataset kosong atau tidak ada wajah yang berhasil dilatih."
        }), 500

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File gambar tidak valid."}), 400
        
    try:
        # Membaca buffer gambar dari request HTTP
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal membaca format gambar yang diunggah."}), 400
            
        # 1 & 2: Deteksi dan Potong Wajah
        cropped_face = detect_and_crop(img)
        if cropped_face is None:
            return jsonify({"error": "Wajah tidak terdeteksi, silakan coba foto lain."}), 400
            
        # 1 (Lanjutan): Preprocessing (Grayscale, Resize 100x100, Normalisasi, Flatten ke 10,000 dim)
        flat_face = preprocess_face(cropped_face)
        
        # 3. Proyeksi ke ruang PCA (Transformasi SVD ke <50 komponen utama)
        face_pca = pca_model.transform([flat_face])
        
        # 4. Kalkulasi Cosine Similarity terhadap dataset
        similarities = cosine_similarity(face_pca, X_train_pca)[0]
        
        best_match_idx = np.argmax(similarities)
        highest_score = similarities[best_match_idx]
        
        # Evaluasi terhadap ambang batas (Threshold 0.80)
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
        return jsonify({"error": f"Terjadi kesalahan internal pada server: {str(e)}"}), 500

# Endpoint API khusus Vercel Serverless Function eksekusi
if __name__ == '__main__':
    app.run(debug=True)