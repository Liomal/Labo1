import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


base_dir = r'C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10'
train_images_path = os.path.join(base_dir, 'train')
val_images_path   = os.path.join(base_dir, 'test')
train_labels_file = os.path.join(base_dir, 'training_labels.csv')
val_labels_file   = os.path.join(base_dir, 'test_labels.csv')

# Función para cargar y aplanar imágenes
def load_images_flat(folder_path, df):
    images = []
    for img_name in tqdm(df['image_name'], desc=f"Cargando {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.flatten() / 255.0)
    return np.array(images)

# Leer etiquetas desde CSV
print("Leyendo etiquetas de entrenamiento...")
t_train = pd.read_csv(train_labels_file)
print("Leyendo etiquetas de validación...")
t_val = pd.read_csv(val_labels_file)

# Cargar imágenes de entrenamiento
print("Cargando imágenes de entrenamiento...")
X_train = load_images_flat(train_images_path, t_train)
y_train = t_train['label'].values

# Cargar imágenes de validación
print("Cargando imágenes de validación...")
X_val = load_images_flat(val_images_path, t_val)
y_val = t_val['label'].values

# Codificar etiquetas
le = LabelEncoder()
y_train_idx = le.fit_transform(y_train)
y_val_idx = le.transform(y_val)

# Entrenar Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=30, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train_idx)

# Evaluar
train_acc = rf.score(X_train, y_train_idx)
val_acc = rf.score(X_val, y_val_idx)
y_pred = rf.predict(X_val)
confm = confusion_matrix(y_val_idx, y_pred)
report = classification_report(y_val_idx, y_pred, target_names=le.classes_)

print(f"Train Acc: {train_acc:.4f}")
print(f"Val Acc:   {val_acc:.4f}")
print("Classification Report:\n", report)

# Guardar resultados en archivos locales
with open("rf_metrics.txt", "w") as f:
    f.write(f"Train Acc: {train_acc:.4f}\nVal Acc: {val_acc:.4f}\n")
    f.write(report)

np.save("rf_confmat.npy", confm)
np.save("rf_classes.npy", le.classes_)