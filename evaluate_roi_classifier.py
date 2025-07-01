# evaluate_roi_classifier.py

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from roi_classifier_dataset import ROICropClassifierDataset
from predict_roi_class import ROIClassifier, CLASS_NAMES


def evaluate_roi_classifier(image_root="dataset", mask_root="mask", log_fn=print):
    clf = ROIClassifier()
    y_true = []
    y_pred = []

    dataset = ROICropClassifierDataset(image_root, mask_root)
    class_map = dataset.class_map
    reverse_map = {v: k for k, v in class_map.items()}

    for img_path, mask_path, class_id in dataset.samples:
        true_label = reverse_map[class_id]
        pred_label = clf.predict(img_path, mask_path)
        y_true.append(true_label)
        y_pred.append(pred_label)

    log_fn("\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    log_fn(report)

    log_fn("📉 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    log_fn("✅ Saved confusion_matrix.png")

if __name__ == "__main__":
    evaluate_roi_classifier()
