# train_roi_classifier.py
"""
Trains a ResNet-based classifier for identifying defect types from ROI images.

This script defines the model architecture, training loop, and functions for
saving the trained model in both PyTorch (.pth) and ONNX formats.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from roi_classifier_dataset import ROICropClassifierDataset
import os


class ResNetWithFeatures(nn.Module):
    """A ResNet model modified to accept additional features.

    This model uses a ResNet18 backbone to extract features from an image and
    concatenates them with a feature vector before passing them to a final
    fully connected layer.
    """
    def __init__(self, num_classes, num_features=4):
        """
        Initializes the ResNetWithFeatures model.

        Args:
            num_classes (int): The number of output classes.
            num_features (int): The number of additional features to concatenate.
        """
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs + num_features, num_classes)

    def forward(self, x, feats):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input image tensor.
            feats (torch.Tensor): The additional feature tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, feats], dim=1)
        return self.fc(x)


class ROIResNetWithFeatures(nn.Module):
    """A ResNet model for ROI classification with additional features.

    Similar to ResNetWithFeatures, but adapted for the ROI classification task.
    """
    def __init__(self, num_classes, feature_dim):
        """Initializes the ROIResNetWithFeatures model.

        Args:
            num_classes: The number of output classes.
            feature_dim: The dimension of the feature vector.
        """
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.classifier = nn.Linear(num_ftrs + feature_dim, num_classes)

    def forward(self, x, feats):
        """Defines the forward pass of the model."""
        x = self.cnn(x)
        x = torch.cat([x, feats], dim=1)
        return self.classifier(x)

def train_roi_classifier(image_root="dataset", mask_root=None, log_fn=print):
    """Trains the ROI classifier and saves the model.

    Args:
        image_root: The root directory of the dataset.
        mask_root: The directory containing the masks. If None, it is
          assumed to be a sub-directory of `image_root` named "Mask".
        log_fn: A function to use for logging progress.
    """
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

    # Export to ONNX format
    dummy_img = torch.randn(1, 1, 224, 224).to(device)
    dummy_feats = torch.randn(1, 8).to(device)
    torch.onnx.export(
        model, (dummy_img, dummy_feats), "roi_classifier.onnx",
        input_names=["image", "feats"], output_names=["output"]
    )
    log_fn("✅ Saved: roi_classifier.onnx")

def run_train_from_ui(log_fn):
    """A wrapper for `train_roi_classifier` to be called from the UI."""
    train_roi_classifier(log_fn=log_fn)

if __name__ == "__main__":
    train_roi_classifier()
