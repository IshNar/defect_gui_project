# dataset_saver.py
import os
import cv2
from datetime import datetime

def save_roi(roi_img, label):
    folder = os.path.join("dataset", label)
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    cv2.imwrite(os.path.join(folder, filename), roi_img)



