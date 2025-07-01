# predict_roi_class.py

import os
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from roi_classifier_dataset import ROICropClassifierDataset

CLASS_NAMES = ["Scratch", "Dent", "Dust"]

class ROIClassifier:
    def __init__(self, weight_path="roi_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        
        # ✅ 파일 존재 확인 + 안전한 로딩
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model file not found: {weight_path}")
        
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = ROICropClassifierDataset("dataset", "mask").transform

    def predict(self, image_path, mask_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "No Defect"

        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
        tensor = self.transform(roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred_class = output.argmax(dim=1).item()
            return CLASS_NAMES[pred_class]

if __name__ == "__main__":
    clf = ROIClassifier()
    label = clf.predict("dataset/Scratch/Dust10.png", "mask/Scratch/Dust10_mask.png")
    print("Predicted:", label)
