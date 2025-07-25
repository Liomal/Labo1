import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

# 1. Cargar el CSV de labels
base_dir      = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
labels_path   = os.path.join(base_dir, "training_labels.csv")
labels_df     = pd.read_csv(labels_path)

# 2. Contar imágenes por clase
counts = labels_df["label"].value_counts().sort_index()
print("Número de imágenes por clase:\n", counts)

# 3. Separar train/validation de forma estratificada
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df["label"],
    random_state=42
)
print(f"\nTraining: {len(train_df)} imágenes\nValidation: {len(val_df)} imágenes")

# 4. Visualizar N ejemplos por clase
n_examples = 3
classes    = sorted(labels_df["label"].unique())
fig, axes = plt.subplots(len(classes), n_examples, figsize=(n_examples*2, len(classes)*2))
fig.suptitle("Ejemplos por clase", y=0.92)

for i, cls in enumerate(classes):
    # Filtrar filenames de esa clase y tomar los primeros n_examples
    fnames = train_df[train_df["label"] == cls]["image_name"].tolist()[:n_examples]
    for j, fname in enumerate(fnames):
        img = Image.open(os.path.join(base_dir, "train", fname))
        ax  = axes[i, j]
        ax.imshow(img)
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(cls, rotation=0, labelpad=40, va="center")

plt.tight_layout()
plt.show()
