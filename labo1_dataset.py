import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    Dataset personalizado que puede recibir labels_path o directamente labels_df.
    """
    def __init__(self,
                 folder_path: str,
                 label_to_idx_map: dict,
                 transforms=None,
                 labels_path: str = None,
                 labels_df: pd.DataFrame = None):
        self.folder_path      = folder_path
        self.transforms       = transforms
        self.label_to_idx_map = label_to_idx_map

        if labels_df is not None:
            self.labels_df = labels_df.reset_index(drop=True)
        elif labels_path is not None:
            self.labels_df = pd.read_csv(labels_path)
        else:
            raise ValueError("Tienes que pasar labels_path o labels_df")

    def __getitem__(self, index):
        row        = self.labels_df.iloc[index]
        image_name = row['image_name']
        label_name = row['label']

        image_path = os.path.join(self.folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transforms:
            image = self.transforms(image)

        label = self.label_to_idx_map[label_name]
        return image, label

    def __len__(self):
        return len(self.labels_df)
