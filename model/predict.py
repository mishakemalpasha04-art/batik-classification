import os
import json
import numpy as np
import joblib
from tkinter import Tk, filedialog
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Load Model dan Label Encoder ===
model_path = 'D:/Batik/model/knn_model.pkl'
label_path = 'D:/Batik/model/label.json'

model = joblib.load(model_path)
with open(label_path, 'r') as f:
    label_map = json.load(f)

# Balikkan mapping: {0: 'batik', 1: 'non_batik', ...}
id_to_label = {int(k): v for k, v in label_map.items()}

# === Ekstraktor fitur (EfficientNetB0) ===
feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

# === Pilih gambar lewat file explorer ===
Tk().withdraw()  # Sembunyikan jendela utama tkinter
file_path = filedialog.askopenfilename(title="Pilih gambar untuk prediksi",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if file_path:
    print(f"\nGambar dipilih: {file_path}")
    features = extract_features(file_path)
    prediction = model.predict([features])[0]
    label = id_to_label[int(prediction)]
    print(f"Prediksi: {label}")
else:
    print("Tidak ada gambar dipilih.")
