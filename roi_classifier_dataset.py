# roi_classifier_dataset.py

import os
import math
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

# Fixed order of classes used across training and inference
CLASS_NAMES = ["Scratch", "Dent", "Dust"]



class ROICropClassifierDataset(Dataset):
    def __init__(self, image_root, mask_root, target_size=(224, 224)):
        self.samples = []
        self.class_map = {}  # folder name to class ID
        self.target_size = target_size

        # Ignore the "Mask" folder which stores segmentation masks only
        #기존의 class_forders 부분은 내림차순으로 정렬해 해당 class indexing을 잡는데 실제 CLASS_NAMES는 다른 순서여서 학습 후에도 결과가 달랐던 것..
        # class_folders = [d for d in sorted(os.listdir(image_root))
        #                  if os.path.isdir(os.path.join(image_root, d)) and d != "Mask"]
        # Folders containing class images
         # Folders containing class images
        class_folders = [d for d in os.listdir(image_root)
                         if os.path.isdir(os.path.join(image_root, d)) and d != "Mask"]
        
        
        # Assign indices following CLASS_NAMES order
        for idx, cls in enumerate(CLASS_NAMES):
            if cls not in class_folders:
                continue
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
        mask_roi = mask[y:y+h, x:x+w]

        # # Shape features
        # area_ratio = cv2.contourArea(largest) / float(w * h)
        # aspect_ratio = float(w) / h if h > 0 else 0.0

        # # Brightness features
        # brightness_mean = roi.mean() / 255.0
        # brightness_std = roi.std() / 255.0

        roi = cv2.resize(roi, self.target_size)

        # Shape features
        area_ratio = cv2.contourArea(largest) / float(w * h) if w * h > 0 else 0.0
        aspect_ratio = float(w) / h if h > 0 else 0.0

        if len(largest) >= 5:
            (center, axes, angle) = cv2.fitEllipse(largest)
            major_axis = max(axes)
            minor_axis = min(axes)
        else:
            major_axis = float(max(w, h))
            minor_axis = float(min(w, h))

        elongation = major_axis / minor_axis if minor_axis > 0 else 0.0
        perimeter = cv2.arcLength(largest, True)
        area = cv2.contourArea(largest)
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

        brightness_mean = roi.mean() / 255.0
        brightness_std = roi.std() / 255.0

        features = np.array([
            area_ratio,
            aspect_ratio,
            brightness_mean,
            brightness_std,
            major_axis,
            minor_axis,
            elongation,
            circularity,
        ], dtype=np.float32)

        
        tensor = self.transform(roi)
        feature_tensor = torch.from_numpy(features)

        return tensor, feature_tensor, class_id
