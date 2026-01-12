import os
import pandas as pd
import torch
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class SVHNCustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_len=5, pad_token=10):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        # Converte string "[4, 7, 8]" para lista [4, 7, 8]
        digits_list = ast.literal_eval(row['digits'])
        
        # Padding
        current_len = len(digits_list)
        if current_len < self.max_len:
            digits_list += [self.pad_token] * (self.max_len - current_len)
        else:
            digits_list = digits_list[:self.max_len]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(digits_list, dtype=torch.long), torch.tensor(current_len)

def get_svhn_loaders(train_csv, train_dir, 
                    test_csv, test_dir, 
                    extra_csv=None, extra_dir=None,
                    batch_size=16, img_size=(640, 640)):
    """
    Função auxiliar para instanciar os dataloaders rapidamente.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.439, 0.434, 0.439], std=[0.204, 0.208, 0.208])
    ])

    train_ds = SVHNCustomDataset(train_csv, train_dir, transform=transform)
    test_ds = SVHNCustomDataset(test_csv, test_dir, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    extra_loader = None
    if extra_csv and extra_dir:
        extra_ds = SVHNCustomDataset(extra_csv, extra_dir, transform=transform)
        extra_loader = DataLoader(extra_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader, extra_loader

def calculate_dataset_stats(dataloader):
    """
    Calcula a média e o desvio padrão por canal (R, G, B) de um dataloader.
    """
    sum_ = torch.tensor([0.0, 0.0, 0.0])
    sum_sq_ = torch.tensor([0.0, 0.0, 0.0])
    total_pixels = 0

    print("Calculando estatísticas do dataset...")
    for images, _, _ in tqdm(dataloader):
        # images shape: [batch_size, 3, height, width]
        batch_size = images.size(0)
        num_pixels = images.size(2) * images.size(3)
        
        # Soma dos valores dos pixels por canal
        sum_ += torch.sum(images, dim=[0, 2, 3])
        
        # Soma dos quadrados dos valores dos pixels por canal (para variância)
        sum_sq_ += torch.sum(images**2, dim=[0, 2, 3])
        
        total_pixels += batch_size * num_pixels

    # Média final
    mean = sum_ / total_pixels
    
    # Desvio padrão final: sqrt( E[X^2] - (E[X])^2 )
    var = (sum_sq_ / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std