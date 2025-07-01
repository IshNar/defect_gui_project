# trainer.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def train_model(log_fn):
    IMG_SIZE = 224
    NUM_CLASSES = 3
    EPOCHS = 50
    BATCH_SIZE = 16
    MODEL_PATH = "model/defect_classifier.onnx"

    log_fn("📦 Training started...")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])


    dataset = ImageFolder("dataset", transform=transform)
    if len(dataset) == 0:
        log_fn("❌ No training data found.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 3)
    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


    # ONNX 저장
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    os.makedirs("model", exist_ok=True)
    torch.onnx.export(model, dummy_input, MODEL_PATH, input_names=["input"], output_names=["output"])
    log_fn(f"✅ Model saved to {MODEL_PATH}")
