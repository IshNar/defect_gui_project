# train_roi_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from roi_classifier_dataset import ROICropClassifierDataset
import os


class ResNetWithFeatures(nn.Module):
    def __init__(self, num_classes, num_features=4):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs + num_features, num_classes)

    def forward(self, x, feats):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, feats], dim=1)
        return self.fc(x)


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

def train_roi_classifier(image_root="dataset", mask_root=None, log_fn=print):
    if mask_root is None:
        mask_root = os.path.join(image_root, "Mask")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ROICropClassifierDataset(image_root, mask_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(dataset.class_map)
    feature_dim = 8
    model = ROIResNetWithFeatures(num_classes, feature_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    log_fn(f"🧠 Training ROI classifier ({num_classes} classes) ...")

    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, feats, labels in loader:
            images, feats, labels = images.to(device), feats.to(device), labels.to(device)
            outputs = model(images, feats)
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

    #ONNX 저장(외부 export)
    dummy_input = torch.randn(1, 1, 244, 244).to(device)
    torch.onnx.export(model, dummy_input, "roi_classifier.onnx",
                  input_names=["input"], output_names=["output"])
    log_fn("✅ Saved: roi_classifier.onnx")

def run_train_from_ui(log_fn):
    train_roi_classifier(log_fn=log_fn)

if __name__ == "__main__":
    train_roi_classifier()
