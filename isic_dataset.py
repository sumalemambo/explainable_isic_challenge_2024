# Clase Dataloader custom para KFolds
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class KFoldISICDataset(Dataset):
    def __init__(self, dataframe, image_path_col, target_col, transforms=None):
        self.df = dataframe
        self.labels = self.df[target_col].values
        self.image_paths = self.df[image_path_col].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Path y target del item
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        image = Image.open(image_path).convert("RGB")

        # Aplicar transformaciones
        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label)