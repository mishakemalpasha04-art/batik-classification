# ini percobaan commit kedua

from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import joblib
import json
from pathlib import Path
from PIL import Image
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = Path(__file__).parent / 'model' / 'knn_model.pkl'
LABEL_PATH = Path(__file__).parent / 'model' / 'label.json'

# Deskripsi batik - gunakan label yang sudah rapi
deskripsi_batik = {
    "Batik Cap Asem Arang": "Batik ini memiliki motif khas yang terinspirasi dari bentuk daun asem dan pola arang yang bertekstur. Umumnya menggunakan warna-warna gelap dengan sentuhan natural.",
    
    "Batik Cap Asem Sinom": "Motif batik ini menampilkan bentuk daun asem muda (sinom) yang melambangkan kesegaran, pertumbuhan, dan semangat baru. Biasanya memiliki warna hijau atau kuning muda.",
    
    "Batik Cap Asem Warak": "Perpaduan motif daun asem dengan unsur mitologis 'warak', makhluk khas budaya Semarang. Batik ini mencerminkan keberagaman dan warisan budaya lokal.",
    
    "Batik Cap Blekok": "Terinspirasi dari burung blekok (jenis bangau), batik ini sering menampilkan siluet burung dengan latar alam rawa. Memberikan kesan tenang dan alami.",
    
    "Batik Cap Blekok Warak": "Gabungan dua ikon lokal: burung blekok dan makhluk warak. Batik ini unik karena menggabungkan unsur fauna dan mitos dalam satu komposisi harmonis.",
    
    "Batik Cap Gambang Semarangan": "Batik dengan motif alat musik tradisional 'gambang' yang berasal dari Semarang. Biasanya dipakai dalam konteks seni pertunjukan atau acara budaya.",
    
    "Batik Cap Kembang Sepatu": "Mengusung motif bunga kembang sepatu yang merepresentasikan keanggunan dan kecantikan tropis. Corak ini populer karena bentuknya yang simetris dan indah.",
    
    "Batik Cap Semarangan": "Motif khas daerah Semarang, biasanya menampilkan elemen landmark, budaya lokal, dan simbol-simbol kota. Menjadi identitas visual yang kuat untuk warga lokal.",
    
    "Batik Cap Tugu Muda": "Motif ini menggambarkan Tugu Muda, ikon perjuangan di kota Semarang. Cocok untuk memperingati nilai sejarah dan semangat kebangsaan.",
    
    "Batik Cap Warak Beras Utah": "Motif ini menggabungkan simbol 'warak' dengan pola 'beras utah' (beras yang tercecer), melambangkan keberkahan dan kelimpahan dalam budaya Jawa."
}


IMG_SIZE = (224, 224)

# Load model dan label
model = joblib.load(MODEL_PATH)

with open(LABEL_PATH, 'r') as f:
    label_dict = json.load(f)

label_dict = {int(k): v for k, v in label_dict.items()}

# Ekstraktor fitur
feature_extractor = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(224, 224, 3)
)

def extract_feature(image):
    image = image.resize(IMG_SIZE).convert('RGB')
    img_array = img_to_array(image)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    features = feature_extractor.predict(img_array, verbose=0).flatten()
    return features.reshape(1, -1)

last_error = None

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    global last_error
    error = last_error
    last_error = None
    return render_template('predict.html', last_error=error)

@app.route('/predict', methods=['POST'])
def predict_result():
    global last_error
    last_error = None

    if 'file' not in request.files:
        last_error = "Tidak ada file yang dikirim."
        return redirect(url_for('predict_page'))

    file = request.files['file']
    if not file.filename:
        last_error = "Nama file kosong."
        return redirect(url_for('predict_page'))

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(save_path)

        image = Image.open(save_path)
        features = extract_feature(image)
        prediction_index = model.predict(features)[0]
        label_key = label_dict.get(prediction_index, f"Label tidak diketahui: {prediction_index}")

        prediction_label = label_dict.get(prediction_index, label_key.title())

        description = deskripsi_batik.get(prediction_label, "Deskripsi tidak tersedia.")
        return render_template('hasil.html', prediction=prediction_label, description=description, preview_path=f'uploads/{filename}')

    except Exception as e:
        last_error = f"Terjadi kesalahan: {str(e)}"
        return redirect(url_for('predict_page'))

if __name__ == '__main__':
    app.run(debug=True)
