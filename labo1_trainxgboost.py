import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class TQDMCallback(TrainingCallback):
    """Callback para mostrar una barra de progreso con tqdm."""
    def __init__(self, total_rounds: int):
        self.pbar = tqdm(total=total_rounds, desc="Boosting Rounds")
    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        self.pbar.update(1)
        return False  # False = continuar entrenamiento
    def after_training(self, model):
        self.pbar.close()
        return model

def load_images_flat(folder_path, df):
    images = []
    for img_name in tqdm(df['image_name'], desc=f"Cargando {os.path.basename(folder_path)}"):
        img = cv2.imread(os.path.join(folder_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.flatten() / 255.0)
    return np.array(images)

def main():
    # --- RUTAS ---
    base_dir = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
    train_images_path = os.path.join(base_dir, 'train')
    val_images_path   = os.path.join(base_dir, 'test')
    train_labels_file = os.path.join(base_dir, 'training_labels.csv')
    val_labels_file   = os.path.join(base_dir, 'test_labels.csv')

    # --- Leer etiquetas ---
    train_df = pd.read_csv(train_labels_file)
    val_df   = pd.read_csv(val_labels_file)

    # --- Cargar y aplanar imágenes ---
    X_train = load_images_flat(train_images_path, train_df)
    y_train = train_df['label'].values
    X_val   = load_images_flat(val_images_path, val_df)
    y_val   = val_df['label'].values

    # --- Codificar etiquetas ---
    le = LabelEncoder()
    y_train_idx = le.fit_transform(y_train)
    y_val_idx   = le.transform(y_val)

    # --- Parámetros y número de rounds ---
    n_rounds = 100
    params = {
        'tree_method':      'gpu_hist',      # acelerar en GPU
        'predictor':        'gpu_predictor',
        'learning_rate':    0.1,
        'max_depth':        6,
        'objective':        'multi:softprob',
        'eval_metric':      'mlogloss',
        'random_state':     42
    }

    # --- Entrenamiento con barra de progreso ---
    print("Entrenando XGBoost con barra de progreso...")
    clf = XGBClassifier(
        n_estimators=n_rounds,
        **params,
        use_label_encoder=False
        # no ponemos callbacks aquí para evitar warning; los pasaremos con set_params
    )
    # Es mejor pasar callbacks como parámetro del modelo según la doc de sklearn API :contentReference[oaicite:0]{index=0}
    clf.set_params(callbacks=[TQDMCallback(n_rounds)])
    clf.fit(X_train, y_train_idx, eval_set=[(X_val, y_val_idx)], verbose=False)

    # --- Evaluación ---
    train_acc = clf.score(X_train, y_train_idx)
    val_acc   = clf.score(X_val,   y_val_idx)
    y_pred    = clf.predict(X_val)
    confm     = confusion_matrix(y_val_idx, y_pred)
    report    = classification_report(y_val_idx, y_pred, target_names=le.classes_)

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")
    print("Classification Report:\n", report)

    # --- Guardar resultados ---
    os.makedirs("results/XGBoost_TQDM", exist_ok=True)
    with open("results/XGBoost_TQDM/metrics.txt", "w") as f:
        f.write(f"Train Acc: {train_acc:.4f}\nVal Acc: {val_acc:.4f}\n")
        f.write(report)
    np.save("results/XGBoost_TQDM/confusion_matrix.npy", confm)
    np.save("results/XGBoost_TQDM/classes.npy", le.classes_)

if __name__ == "__main__":
    main()
