# predict_roi_class.py

import os
import math
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from roi_classifier_dataset import ROICropClassifierDataset, CLASS_NAMES

# CLASS_NAMES = ["Scratch", "Dent", "Dust"]


class ROIResNetWithFeatures(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.classifier = nn.Linear(num_ftrs + feature_dim, num_classes)

    def forward(self, x, feats):
        x = self.cnn(x)
        x = torch.cat([x, feats], dim=1)
        return self.classifier(x)



class ROIClassifier:
    def __init__(self, weight_path="roi_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ROIResNetWithFeatures(len(CLASS_NAMES), feature_dim=8)
        # ✅ 파일 존재 확인 + 안전한 로딩
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model file not found: {weight_path}")
        
        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = ROICropClassifierDataset("dataset", os.path.join("dataset", "Mask")).transform

    def predict(self, image_path, mask_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        # 이미지 크기 확인 후 resize
        if img.shape[0] < mask.shape[0] or img.shape[1] < mask.shape[1]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)


        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "No Defect"

        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        area_ratio = cv2.contourArea(largest) / float(w * h)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        brightness_mean = roi.mean() / 255.0
        brightness_std = roi.std() / 255.0

        roi = cv2.resize(roi, (224, 224))

        area_ratio = cv2.contourArea(largest) / float(w * h) if w * h > 0 else 0.0
        aspect_ratio = float(w) / h if h > 0 else 0.0

        if len(largest) >= 5:
            (_, axes, _) = cv2.fitEllipse(largest)
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


        tensor = self.transform(roi).unsqueeze(0).to(self.device)
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor, feat_tensor)
            pred_class = output.argmax(dim=1).item()
            return CLASS_NAMES[pred_class]

if __name__ == "__main__":
    clf = ROIClassifier()
    label = clf.predict("dataset/Scratch/Dust10.png", os.path.join("dataset", "Mask", "Dust10_mask.png"))
    print("Predicted:", label)
