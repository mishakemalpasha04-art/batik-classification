import os
import time
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# === Path Dataset ===
base_path = 'D:\skripsi\Batik\dataset'
splits = [f"split_{i}" for i in range(1, 4)]

# === Ekstraktor Fitur ===
feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

# === Load Dataset ===
def load_dataset(split_path, subset):
    X, y = [], []
    subset_path = os.path.join(split_path, subset)
    for class_name in os.listdir(subset_path):
        class_path = os.path.join(subset_path, class_name)
        for fname in tqdm(os.listdir(class_path), desc=f"{subset.upper()} - {class_name}"):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    path = os.path.join(class_path, fname)
                    feat = extract_features(path)
                    X.append(feat)
                    y.append(class_name)
                except:
                    continue
    return np.array(X), np.array(y)

# === Grid Search Parameter ===
param_grid = {
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# === Training Loop ===
start_global = time.time()
all_results = []
best_model = None
best_acc = 0
best_model_info = {}
best_le = None

for split in splits:
    print(f"\n=== Proses {split.upper()} ===")
    split_path = os.path.join(base_path, split)

    X_train, y_train = load_dataset(split_path, 'train')
    X_test, y_test = load_dataset(split_path, 'test')

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train_enc)

    # Evaluasi di test set
    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test_enc, y_pred)
    print(f"Akurasi uji terbaik: {test_acc:.4f} (k={grid.best_params_['n_neighbors']}, {grid.best_params_['weights']}, {grid.best_params_['metric']})")

    # Simpan semua hasil dan tambahkan test_acc hanya untuk best param
    for i, params in enumerate(grid.cv_results_['params']):
        acc = grid.cv_results_['mean_test_score'][i]
        is_best = (params == grid.best_params_)
        all_results.append({
            'split': split,
            'k': params['n_neighbors'],
            'weights': params['weights'],
            'metric': params['metric'],
            'cv_accuracy': acc,
            'test_accuracy': test_acc if is_best else None
        })

    if test_acc > best_acc:
        best_acc = test_acc
        best_model = grid.best_estimator_
        best_model_info = {
            'split': split,
            'k': grid.best_params_['n_neighbors'],
            'weights': grid.best_params_['weights'],
            'metric': grid.best_params_['metric']
        }
        best_le = le

# === Simpan Model Terbaik ===
if best_model:
    os.makedirs('D:/Batik/model', exist_ok=True)
    joblib.dump(best_model, 'D:/Batik/model/knn_model.pkl')
    with open('D:/Batik/model/label.json', 'w') as f:
        label_dict = {str(k): v for k, v in zip(best_le.transform(best_le.classes_), best_le.classes_)}
        json.dump(label_dict, f)
    print("\nModel terbaik disimpan ke knn_model.pkl")
    print(f"Split terbaik: {best_model_info['split']}")
    print(f"Akurasi terbaik: {best_acc:.4f}")
    print(f"Param: k={best_model_info['k']}, weights={best_model_info['weights']}, metric={best_model_info['metric']}")

# === Simpan CSV ===
df = pd.DataFrame(all_results)
df.to_csv('D:/Batik/model/hasil.csv', index=False)
print("Hasil disimpan ke hasil.csv")

# === Visualisasi ===
plt.figure(figsize=(14, 6))
for split in splits:
    subset = df[df['split'] == split]
    for weight in ['uniform', 'distance']:
        for metric in ['euclidean', 'manhattan']:
            line = subset[(subset['weights'] == weight) & (subset['metric'] == metric)]
            if not line.empty:
                plt.plot(line['k'], line['cv_accuracy'], label=f"{split} - {weight} - {metric}")

plt.title("Akurasi KNN per Split dan Parameter (EfficientNet Features)")
plt.xlabel("Nilai k")
plt.ylabel("CV Akurasi")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

print(f"\nTotal waktu eksekusi: {time.time() - start_global:.2f} detik")
