# labo1_traincnn_both.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from labo1_model import SimpleCNN
from labo1_dataset import CustomDataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark     = False

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train_and_validate(model, optimizer, scheduler,
                       train_loader, val_loader,
                       criterion, nb_epochs,
                       results_dir, device):
    os.makedirs(results_dir, exist_ok=True)
    best_val_acc = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, nb_epochs+1):
        # TRAIN
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

        # VALIDATION
        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{nb_epochs} [Val]"):
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

        print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, "
              f"Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # scheduler step
        scheduler.step(val_loss)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, os.path.join(results_dir, f'best_model_{optimizer.__class__.__name__}.pth'))
            print(f"→ Best model saved (val_acc={best_val_acc:.4f})")

    # save metrics
    np.save(os.path.join(results_dir, 'train_loss.npy'), np.array(metrics['train_loss']))
    np.save(os.path.join(results_dir, 'val_loss.npy'),   np.array(metrics['val_loss']))
    np.save(os.path.join(results_dir, 'train_acc.npy'),  np.array(metrics['train_acc']))
    np.save(os.path.join(results_dir, 'val_acc.npy'),    np.array(metrics['val_acc']))

    return metrics

def main():
    set_seed(42)

    # paths
    base_dir     = r"C:\Users\USUARIO\OneDrive\Escritorio\Labs\Datasets\CIFAR-10"
    train_folder = os.path.join(base_dir, "train")
    labels_csv   = os.path.join(base_dir, "training_labels.csv")

    # hyperparams
    batch_size = 32
    img_size   = (32, 32)
    nb_epochs  = 10
    test_size  = 0.2

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])

    # read & split
    df = pd.read_csv(labels_csv)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=test_size,
        stratify=df["label"], random_state=42
    )

    # label map
    labels    = sorted(df["label"].unique())
    label2idx = {lab: i for i, lab in enumerate(labels)}

    # datasets
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
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("Usando dispositivo:", device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # loop over optimizers
    optimizers = {
        'SGD':  optim.SGD,
        'Adam': optim.Adam
    }

    for opt_name, OptClass in optimizers.items():
        print(f"\n===== Entrenando con {opt_name} =====")
        set_seed(42)

        # model and reset weights
        model = SimpleCNN(nb_classes=len(labels), img_size=img_size).to(device)
        model.apply(reset_weights)

        # optimizer settings
        if opt_name == 'SGD':
            optimizer = OptClass(model.parameters(),
                                 lr=0.01,
                                 momentum=0.9,
                                 weight_decay=1e-4)
        else:  # Adam
            optimizer = OptClass(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=0.0)

        # scheduler
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.1,
                                      patience=3)

        # results directory
        results_dir = os.path.join("results", opt_name)
        metrics = train_and_validate(model, optimizer, scheduler,
                                     train_loader, val_loader,
                                     criterion, nb_epochs,
                                     results_dir, device)

        # plot curves
        epochs = np.arange(1, nb_epochs+1)
        plt.figure()
        plt.plot(epochs, metrics['train_loss'], marker='o', label='Train Loss')
        plt.plot(epochs, metrics['val_loss'],   marker='o', label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{opt_name} Loss')
        plt.legend(); plt.tight_layout(); plt.show()

        plt.figure()
        plt.plot(epochs, metrics['train_acc'], marker='o', label='Train Acc')
        plt.plot(epochs, metrics['val_acc'],   marker='o', label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'{opt_name} Accuracy')
        plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
