import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# (Opcional) mejora el estilo de las gráficas
sns.set()

# --------------------------------------------------
# 1) Ajusta aquí la ruta base de tu proyecto
# --------------------------------------------------
base_dir = r'C:\Users\USUARIO\OneDrive\Escritorio\Labo1'
cnn_dir  = os.path.join(base_dir, 'results')

# --------------------------------------------------
# 2) Rutas de los archivos de resultados
# --------------------------------------------------

# Random Forest (están directamente en Labo1)
rf_metrics_path  = os.path.join(base_dir, 'rf_metrics')       # sin extensión .txt
rf_confmat_path  = os.path.join(base_dir, 'rf_confmat.npy')
rf_classes_path  = os.path.join(base_dir, 'rf_classes.npy')

# Simple CNN (están en Labo1\results)
cnn_metrics_path    = os.path.join(cnn_dir,  'cnn_metrics')    # sin extensión .txt
cnn_confmat_path    = os.path.join(cnn_dir,  'cnn_confmat.npy')
cnn_classes_path    = os.path.join(cnn_dir,  'cnn_classes.npy')
cnn_train_loss_path = os.path.join(cnn_dir,  'cnn_train_loss.npy')
cnn_val_loss_path   = os.path.join(cnn_dir,  'cnn_val_loss.npy')
cnn_train_acc_path  = os.path.join(cnn_dir,  'cnn_train_acc.npy')
cnn_val_acc_path    = os.path.join(cnn_dir,  'cnn_val_acc.npy')

# SVM (están directamente en Labo1)
svm_metrics_path = os.path.join(base_dir, 'svm_metrics')       # sin extensión .txt
svm_confmat_path = os.path.join(base_dir, 'svm_confmat.npy')
svm_classes_path = os.path.join(base_dir, 'svm_classes.npy')

# --------------------------------------------------
# 3) Carga de métricas y matrices
# --------------------------------------------------

# Random Forest
with open(rf_metrics_path, 'r') as f:
    lines = f.readlines()
    rf_train_acc = float(lines[0].split(':')[1].strip())
    rf_val_acc   = float(lines[1].split(':')[1].strip())
rf_confmat = np.load(rf_confmat_path)
rf_classes = np.load(rf_classes_path, allow_pickle=True)

# Simple CNN
with open(cnn_metrics_path, 'r') as f:
    lines = f.readlines()
    # Suponemos que guarda al menos la validación
    cnn_val_acc_metric = float(lines[0].split(':')[1].strip())
cnn_confmat       = np.load(cnn_confmat_path)
cnn_classes       = np.load(cnn_classes_path, allow_pickle=True)
cnn_train_loss    = np.load(cnn_train_loss_path)
cnn_val_loss      = np.load(cnn_val_loss_path)
cnn_train_acc_arr = np.load(cnn_train_acc_path)
cnn_val_acc_arr   = np.load(cnn_val_acc_path)

# SVM
with open(svm_metrics_path, 'r') as f:
    lines = f.readlines()
    svm_train_acc = float(lines[0].split(':')[1].strip())
    svm_val_acc   = float(lines[1].split(':')[1].strip())
svm_confmat = np.load(svm_confmat_path)
svm_classes = np.load(svm_classes_path, allow_pickle=True)

# --------------------------------------------------
# 4) Gráfica comparativa de Accuracy
# --------------------------------------------------
modelos   = ['Random Forest', 'Simple CNN', 'SVM']
acc_train = [rf_train_acc, cnn_train_acc_arr[-1], svm_train_acc]
acc_val   = [rf_val_acc,   cnn_val_acc_arr[-1],   svm_val_acc]

x     = np.arange(len(modelos))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, acc_train, width, label='Train', alpha=0.7)
plt.bar(x + width/2, acc_val,   width, label='Validation', alpha=0.9)
plt.xticks(x, modelos)
plt.ylabel('Accuracy')
plt.title('Comparación de Accuracy (Train vs Validation)')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5) Gráficas CNN: Loss y Accuracy por época
# --------------------------------------------------
epochs = np.arange(1, len(cnn_train_loss) + 1)

plt.figure()
plt.plot(epochs, cnn_train_loss,    label='Train Loss')
plt.plot(epochs, cnn_val_loss,      label='Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('CNN - Loss por Época')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(epochs, cnn_train_acc_arr, label='Train Acc')
plt.plot(epochs, cnn_val_acc_arr,   label='Val Acc')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('CNN - Accuracy por Época')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6) Matrices de confusión con heatmap de seaborn
# --------------------------------------------------
plt.figure(figsize=(6,6))
sns.heatmap(rf_confmat, annot=True, fmt='d',
            xticklabels=rf_classes, yticklabels=rf_classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest - Matriz de Confusión')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
sns.heatmap(cnn_confmat, annot=True, fmt='d',
            xticklabels=cnn_classes, yticklabels=cnn_classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN - Matriz de Confusión')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
sns.heatmap(svm_confmat, annot=True, fmt='d',
            xticklabels=svm_classes, yticklabels=svm_classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM - Matriz de Confusión')
plt.tight_layout()
plt.show()

