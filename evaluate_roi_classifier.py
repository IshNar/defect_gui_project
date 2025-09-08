# evaluate_roi_classifier.py
"""
Evaluates the performance of the trained ROI classifier.

This script iterates through the dataset, generates predictions for each sample,
and then computes and displays a classification report and a confusion matrix.
"""
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from roi_classifier_dataset import ROICropClassifierDataset, CLASS_NAMES
from predict_roi_class import ROIClassifier

def evaluate_roi_classifier(image_root="dataset", mask_root=None, log_fn=print):
    """Evaluates the ROI classifier and logs the results.

    Args:
        image_root: The root directory of the dataset.
        mask_root: The directory containing the masks. If None, it is
          assumed to be a sub-directory of `image_root` named "Mask".
        log_fn: A function to use for logging progress and results.
    """
    if mask_root is None:
        mask_root = os.path.join(image_root, "Mask")
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
