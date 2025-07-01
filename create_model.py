# create_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# 하이퍼파라미터
IMG_SIZE = 224
NUM_CLASSES = 3
EPOCHS = 1  # 테스트 목적
BATCH_SIZE = 8

# 간단한 CNN 모델
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
    nn.Linear(128, NUM_CLASSES)
)

# 전처리
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# 데이터셋 로딩
dataset = ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 학습 (1 Epoch만)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ONNX로 저장
dummy_input = torch.randn(1, 1, 224, 224)
os.makedirs("model", exist_ok=True)
torch.onnx.export(model, dummy_input, "model/defect_classifier.onnx",
                  input_names=["input"], output_names=["output"])
print("✅ 모델 저장 완료: model/defect_classifier.onnx")
