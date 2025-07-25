import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Ajusta aquí tu usuario de Windows
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
train_df = pd.read_csv(train_labels_file)
print("Leyendo etiquetas de validación...")
val_df   = pd.read_csv(val_labels_file)

# Subconjunto SVM
train_df = train_df.sample(5000, random_state=42)
val_df   = val_df.sample(1000, random_state=42)

# Cargar imágenes de entrenamiento
print("Cargando imágenes de entrenamiento...")
X_train = load_images_flat(train_images_path, train_df)
y_train = train_df['label'].values

# Cargar imágenes de validación
print("Cargando imágenes de validación...")
X_val   = load_images_flat(val_images_path, val_df)
y_val   = val_df['label'].values

# Codificar etiquetas
le = LabelEncoder()
y_train_idx = le.fit_transform(y_train)
y_val_idx   = le.transform(y_val)

# Entrenar SVM
print("Entrenando SVM...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train_idx)

# Evaluar
print("Evaluando...")
train_acc = svm.score(X_train, y_train_idx)
val_acc   = svm.score(X_val, y_val_idx)
y_pred    = svm.predict(X_val)
confm     = confusion_matrix(y_val_idx, y_pred)
report    = classification_report(y_val_idx, y_pred, target_names=le.classes_)

print(f"Train Acc: {train_acc:.4f}")
print(f"Val Acc:   {val_acc:.4f}")
print("Classification Report:\n", report)

# Guardar resultados en archivos locales
with open("svm_metrics.txt", "w") as f:
    f.write(f"Train Acc: {train_acc:.4f}\nVal Acc: {val_acc:.4f}\n")
    f.write(report)

np.save("svm_confmat.npy", confm)
np.save("svm_classes.npy", le.classes_)
