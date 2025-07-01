# train_roi_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from roi_classifier_dataset import ROICropClassifierDataset
import os

def get_model(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale input
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_roi_classifier(image_root="dataset", mask_root="mask", log_fn=print):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ROICropClassifierDataset(image_root, mask_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(dataset.class_map)
    model = get_model(num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    log_fn(f"🧠 Training ROI classifier ({num_classes} classes) ...")

    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        log_fn(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "roi_classifier.pth")
    log_fn("✅ Saved: roi_classifier.pth")

def run_train_from_ui(log_fn):
    train_roi_classifier(log_fn=log_fn)

if __name__ == "__main__":
    train_roi_classifier()
