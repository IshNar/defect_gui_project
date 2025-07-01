# predict_overlay_viewer.py

import torch
import cv2
import numpy as np
from unet_multiclass_train import UNet

CLASS_NAMES = ["BG", "Scratch", "Dent", "Dust"]
CLASS_COLORS = {
    1: (0, 255, 0),     # Green for Scratch
    2: (0, 0, 255),     # Red for Dent
    3: (255, 255, 0)    # Cyan for Dust
}

def load_model(weight_path="seg_multiclass.pth", num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

def predict_mask(model, device, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig_size = img.shape[::-1]
    img_resized = cv2.resize(img, (224, 224))
    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)  # [H, W]
    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)
    return pred_resized

def overlay_prediction(original_path, pred_mask):
    image = cv2.imread(original_path)
    overlay = image.copy()
    for class_id, color in CLASS_COLORS.items():
        mask = (pred_mask == class_id)
        overlay[mask] = color
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    return blended

def run_overlay_prediction(image_path):
    model, device = load_model("seg_multiclass.pth")
    pred_mask = predict_mask(model, device, image_path)
    result = overlay_prediction(image_path, pred_mask)
    return result
