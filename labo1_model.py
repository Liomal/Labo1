import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, nb_classes, img_size=(32, 32)):
        super(SimpleCNN, self).__init__()
        # --- Bloques convolucionales ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)

        # --- Cálculo del tamaño plano tras el pooling ---
        h, w = img_size
        h, w = h // 2, w // 2  # solo un pooling
        flat_size = 64 * h * w

        # --- Capas densas intermedias ---
        self.fc1        = nn.Linear(flat_size, 128)
        self.dropout1   = nn.Dropout(0.5)
        self.fc2        = nn.Linear(128, 128)
        self.dropout2   = nn.Dropout(0.5)
        # Capa de salida
        self.fc_out     = nn.Linear(128, nb_classes)

    def forward(self, x, debug=False):
        shapes = []

        # Conv1 → ReLU → Pool → Dropout
        shapes.append(x.shape)
        x = F.relu(self.conv1(x)); shapes.append(x.shape)
        x = self.pool(x);           shapes.append(x.shape)

        # Conv2 → ReLU
        x = F.relu(self.conv2(x));  shapes.append(x.shape)
        # Conv3 → ReLU
        x = F.relu(self.conv3(x));  shapes.append(x.shape)

        # Aplanar
        x = x.view(x.size(0), -1);  shapes.append(x.shape)

        # Densas intermedias
        x = F.relu(self.fc1(x));    shapes.append(x.shape)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x));    shapes.append(x.shape)
        x = self.dropout2(x)

        # Salida
        out = self.fc_out(x);       shapes.append(out.shape)

        if debug:
            for i, s in enumerate(shapes):
                print(f"→ Shape after block {i}: {tuple(s)}")
        return out
