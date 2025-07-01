# unet_multiclass_train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".png", "_mask.png"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        image = self.transform(image)
        mask = torch.from_numpy(mask.astype(np.int64))  # no one-hot, int64 class IDs

        return image, mask

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([nn.functional.interpolate(e3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, scale_factor=2), e1], dim=1))
        out = self.final(d1)
        return out  # raw logits (for CrossEntropy)

def train(log_fn):
    NUM_CLASSES = 4  # 0=background, 1=Scratch, 2=Dent, 3=Dust
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegDataset("dataset/Scratch", "dataset/Mask")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    log_fn("🧠 Training Multi-Class Segmentation Model...")

    for epoch in range(50):
        total_loss = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)  # [B, C, H, W]
            loss = criterion(output, mask)  # CrossEntropy expects [B, H, W] target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        log_fn(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "seg_multiclass.pth")
    log_fn("✅ Saved: seg_multiclass.pth")

if __name__ == "__main__":
    train(print)
