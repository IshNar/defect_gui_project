# roi_classifier_dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class ROICropClassifierDataset(Dataset):
    def __init__(self, image_root, mask_root, target_size=(224, 224)):
        self.samples = []
        self.class_map = {}  # folder name to class ID
        self.target_size = target_size

        # Ignore the "Mask" folder which stores segmentation masks only
        class_folders = [d for d in sorted(os.listdir(image_root))
                         if os.path.isdir(os.path.join(image_root, d)) and d != "Mask"]
        
        for idx, cls in enumerate(class_folders):
            self.class_map[cls] = idx
            image_dir = os.path.join(image_root, cls)
            mask_dir = os.path.join(mask_root, cls)

            for fname in os.listdir(image_dir):
                if fname.lower().endswith((".png", ".jpg", ".bmp")):
                    img_path = os.path.join(image_dir, fname)
                    mask_path = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))
                    if not os.path.exists(mask_path):
                        # fallback to a flat mask directory
                        mask_path = os.path.join(mask_root, fname.replace(".png", "_mask.png"))
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, idx))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize grayscale
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, class_id = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        # 이미지 크기 확인 후 resize
        if img.shape[0] < mask.shape[0] or img.shape[1] < mask.shape[1]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Extract largest contour
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise ValueError(f"No valid contour found in mask: {mask_path}")

        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img[y:y+h, x:x+w]

        roi = cv2.resize(roi, self.target_size)
        tensor = self.transform(roi)

        return tensor, class_id
