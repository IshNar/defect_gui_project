# create_model.py
"""
Creates, trains, and exports a simple CNN model for defect classification.

This script defines a simple CNN, trains it for one epoch on the dataset,
and then exports the trained model to ONNX format.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Hyperparameters
IMG_SIZE = 224
NUM_CLASSES = 3
EPOCHS = 1  # For testing purposes
BATCH_SIZE = 8


# A simple CNN model
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
    nn.Linear(128, NUM_CLASSES)
)

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load the dataset
dataset = ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train the model for one epoch
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Export the model to ONNX format
dummy_input = torch.randn(1, 1, 224, 224)
os.makedirs("model", exist_ok=True)
torch.onnx.export(model, dummy_input, "model/defect_classifier.onnx",
                  input_names=["input"], output_names=["output"])
print("✅ Model saved: model/defect_classifier.onnx")
