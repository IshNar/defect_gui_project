# dataset_saver.py
"""Saves ROI images to the dataset directory."""
import os
import cv2
from datetime import datetime

def save_roi(roi_img, label):
    """Saves an ROI image to a labeled sub-directory in the dataset.

    Args:
        roi_img: The ROI image to save (NumPy array).
        label: The class label for the ROI.
    """
    folder = os.path.join("dataset", label)
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    cv2.imwrite(os.path.join(folder, filename), roi_img)



