import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from labo1_dataset import CustomDataset

# ----------------------------------
# Funciones de ayuda a nivel módulo
# ----------------------------------
def set_seed(seed: int = 42):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark     = False

def flatten_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Transformación para aplanar el tensor de imagen.
    x: Tensor de forma [C, H, W]
    devuelve Tensor de forma [C*H*W]
    """
    return x.view(-1)

# ----------------------------------
# Definición de la MLP
# ----------------------------------
class MLP(nn.Module):
    """Red MLP simple: FC → ReLU → Dropout → FC."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ----------------------------------
# Entrenamiento principal
# ----------------------------------
def main():
    set_seed(42)

    # Rutas
    base_dir     = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
    train_folder = os.path.join(base_dir, "train")
    labels_csv   = os.path.join(base_dir, "training_labels.csv")

    # Hiperparámetros
    batch_size = 64
    num_epochs = 10
    lr         = 1e-3
    test_size  = 0.2

    # Transforms: ToTensor y flatten
    flat_tf = transforms.Compose([
        transforms.ToTensor(),         
        transforms.Lambda(flatten_transform)  
    ])

    # Leer CSV y split estratificado
    df = pd.read_csv(labels_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=42
    )

    # Mapeo de etiquetas
    classes   = sorted(df["label"].unique())
    label2idx = {lab: i for i, lab in enumerate(classes)}

    # Datasets
    train_ds = CustomDataset(
        folder_path      = train_folder,
        label_to_idx_map = label2idx,
        transforms       = flat_tf,
        labels_df        = train_df
    )
    val_ds = CustomDataset(
        folder_path      = train_folder,
        label_to_idx_map = label2idx,
        transforms       = flat_tf,
        labels_df        = val_df
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    # Dispositivo 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("Usando dispositivo:", device)

    # Modelo, criterio y optimizador
    input_dim = 3 * 32 * 32
    model     = MLP(input_dim=input_dim, hidden_dim=512, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Historial de métricas
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    # Bucle de entrenamiento y validación
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        running_loss, running_corrects = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss    = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss     += loss.item() * x.size(0)
            running_corrects += (preds == y).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc  = running_corrects / len(train_ds)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss    = criterion(outputs, y)

                preds = outputs.argmax(dim=1)
                running_loss     += loss.item() * x.size(0)
                running_corrects += (preds == y).sum().item()

        val_loss = running_loss / len(val_ds)
        val_acc  = running_corrects / len(val_ds)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: "
              f"Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f} | "
              f"Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

    # Evaluación final
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds   = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm         = confusion_matrix(all_labels, all_preds)
    report     = classification_report(all_labels, all_preds, target_names=classes)
    final_acc  = (all_preds == all_labels).mean()

    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    print("Classification Report:\n", report)

    # Guardar resultados
    results_dir = "results/MLP"
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "mlp_train_loss.npy"), np.array(history["train_loss"]))
    np.save(os.path.join(results_dir, "mlp_val_loss.npy"),   np.array(history["val_loss"]))
    np.save(os.path.join(results_dir, "mlp_train_acc.npy"),  np.array(history["train_acc"]))
    np.save(os.path.join(results_dir, "mlp_val_acc.npy"),    np.array(history["val_acc"]))
    np.save(os.path.join(results_dir, "mlp_confmat.npy"),    cm)
    np.save(os.path.join(results_dir, "mlp_classes.npy"),    np.array(classes))
    with open(os.path.join(results_dir, "mlp_report.txt"), "w") as f:
        f.write(f"Final Validation Accuracy: {final_acc:.4f}\n")
        f.write(report)

    # Plots de curvas
    epochs = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"],   marker="o", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss");     plt.title("MLP Loss");     plt.legend()
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"],   marker="o", label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("MLP Accuracy"); plt.legend()
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
