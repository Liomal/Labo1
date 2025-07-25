import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from labo1_dataset import CustomDataset

def set_seed(seed: int = 42):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark     = False

def train_and_validate(model, optimizer, scheduler,
                       train_loader, val_loader,
                       criterion, nb_epochs,
                       results_dir, device):
    """Bucle de entrenamiento + validación con StepLR y checkpoint."""
    os.makedirs(results_dir, exist_ok=True)
    best_val_acc = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, nb_epochs+1):
        # --- TRAIN ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{nb_epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            running_loss     += loss.item() * x.size(0)
            running_corrects += (preds == y).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = running_corrects / len(train_loader.dataset)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)

        # --- VALIDATION ---
        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{nb_epochs} [ Val ]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                preds = out.argmax(dim=1)
                running_loss     += loss.item() * x.size(0)
                running_corrects += (preds == y).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc  = running_corrects / len(val_loader.dataset)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        print(f"Epoch {epoch}: "
              f"Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, "
              f"Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # StepLR
        scheduler.step()

        # Guardar mejor checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, os.path.join(results_dir, 'best_resnet_checkpoint.pth'))
            print(f"→ Best model saved (val_acc={best_val_acc:.4f})")

    # Guardar métricas
    np.save(os.path.join(results_dir, 'resnet_train_loss.npy'), np.array(metrics['train_loss']))
    np.save(os.path.join(results_dir, 'resnet_val_loss.npy'),   np.array(metrics['val_loss']))
    np.save(os.path.join(results_dir, 'resnet_train_acc.npy'),  np.array(metrics['train_acc']))
    np.save(os.path.join(results_dir, 'resnet_val_acc.npy'),    np.array(metrics['val_acc']))

    return metrics

def main():
    set_seed(42)

    # Rutas
    base_dir      = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
    train_folder  = os.path.join(base_dir, "train")
    labels_csv    = os.path.join(base_dir, "training_labels.csv")

    # Parámetros
    batch_size    = 32
    nb_epochs     = 8
    lr            = 1e-3
    test_size     = 0.2
    num_classes   = 10

    # Transforms (ImageNet stats)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Leer CSV y split estratificado
    df = pd.read_csv(labels_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=42
    )

    # Mapeo etiqueta → índice
    classes   = sorted(df["label"].unique())
    label2idx = {lab: i for i, lab in enumerate(classes)}

    # Datasets y loaders
    train_ds = CustomDataset(
        folder_path      = train_folder,
        label_to_idx_map = label2idx,
        transforms       = train_tf,
        labels_df        = train_df
    )
    val_ds = CustomDataset(
        folder_path      = train_folder,
        label_to_idx_map = label2idx,
        transforms       = val_tf,
        labels_df        = val_df
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("Usando dispositivo:", device)

    # Cargar ResNet-50 preentrenada y adaptar fc
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)
    model       = model.to(device)

    # Congelar todo menos la última capa (opcional)
    #for param in model.parameters():
    #    param.requires_grad = False
    #for param in model.fc.parameters():
    #    param.requires_grad = True

    # Pérdida, optimizador y scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Entrenamiento y validación
    results_dir = "results/ResNet50"
    metrics = train_and_validate(model, optimizer, scheduler,
                                 train_loader, val_loader,
                                 criterion, nb_epochs,
                                 results_dir, device)

    # Plots de curvas
    epochs = np.arange(1, nb_epochs+1)
    plt.figure()
    plt.plot(epochs, metrics['train_loss'], marker='o', label='Train Loss')
    plt.plot(epochs, metrics['val_loss'],   marker='o', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('ResNet50 Loss')
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(epochs, metrics['train_acc'], marker='o', label='Train Acc')
    plt.plot(epochs, metrics['val_acc'],   marker='o', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('ResNet50 Acc')
    plt.legend(); plt.tight_layout(); plt.show()

    # Evaluación final
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Final Eval"):
            x, y = x.to(device), y.to(device)
            out  = model(x)
            preds= out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm         = confusion_matrix(all_labels, all_preds)
    report     = classification_report(all_labels, all_preds, target_names=classes)
    final_acc  = (all_preds == all_labels).mean()

    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    print("Classification Report:\n", report)

    # Guardar confusion matrix y report
    np.save(os.path.join(results_dir, "resnet_confmat.npy"), cm)
    with open(os.path.join(results_dir, "resnet_report.txt"), "w") as f:
        f.write(f"Final Validation Accuracy: {final_acc:.4f}\n")
        f.write(report)

if __name__ == "__main__":
    main()
