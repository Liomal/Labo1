import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from labo1_dataset import CustomDataset
from sklearn.model_selection import train_test_split


# --- Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_dir = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
val_csv = os.path.join(base_dir, "training_labels.csv")
val_dir = os.path.join(base_dir, "train")

# --- Clases ---
classes = sorted(pd.read_csv(val_csv)["label"].unique())
label2idx = {label: idx for idx, label in enumerate(classes)}
idx2label = {idx: label for label, idx in label2idx.items()}

# --- Transforms y dataset ---
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
df = pd.read_csv(val_csv)
_, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

val_dataset = CustomDataset(val_dir, label2idx, transforms=val_tf, labels_df=val_df)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- Modelo cargado ---
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
checkpoint = torch.load("results/ResNet50/best_resnet_checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(device)

# --- Identificar errores ---
incorrect_samples = []

with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1).item()
        true = y.item()
        if pred != true:
            incorrect_samples.append((val_df.iloc[i]["image_name"], pred, true))

# --- Mostrar algunas imágenes mal clasificadas ---
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for idx, (img_name, pred_idx, true_idx) in enumerate(incorrect_samples[:15]):
    img_path = os.path.join(val_dir, img_name)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].axis("off")
    axes[idx].set_title(f"P: {idx2label[pred_idx]}\nT: {idx2label[true_idx]}")

plt.tight_layout()
plt.show()
