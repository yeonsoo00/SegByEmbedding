import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import (
    Compose, Resize, HorizontalFlip, RandomRotate90,
    RandomBrightnessContrast, ShiftScaleRotate
)
from albumentations.pytorch import ToTensorV2

class DualMaskDataset(Dataset):
    def __init__(self, image_dir, mask1_dir, mask2_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask1_dir = mask1_dir
        self.mask2_dir = mask2_dir
        self.image_files = sorted(os.listdir(image_dir))

        self.transform = Compose([
            Resize(*size),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
            RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask1_path = os.path.join(self.mask1_dir, fname)
        mask2_path = os.path.join(self.mask2_dir, fname)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0, 1]
        mask1 = (mask1 > 127).astype(np.float32)
        mask2 = (mask2 > 127).astype(np.float32)

        augmented = self.transform(image=image, masks=[mask1, mask2])
        image = augmented['image']
        mask1, mask2 = augmented['masks']
        masks = torch.stack([mask1, mask2], dim=0)  # Shape: (2, H, W)

        return image, masks


class DualOutputUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 2, 1)  # 2-channel output

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualOutputUNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = DualMaskDataset("dataset/images", "dataset/mask1", "dataset/mask2")
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)  # shape: (B, 2, H, W)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")