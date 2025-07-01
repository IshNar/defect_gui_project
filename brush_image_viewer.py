# brush_image_viewer.py

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2
import os

class BrushImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(512, 512)  # 브러시 작업 공간 크기 (수정 가능)
        self.setStyleSheet("border: 1px solid black")
        self.image = None             # 원본 이미지
        self.mask = None              # 마스크 (uint8: class index)
        self.image_path = None
        self.drawing = False
        self.brush_radius = 8
        self.current_class = 1  # default: 1 = Scratch

    def set_class(self, class_id):
        self.current_class = class_id

    def load_image(self, path):
        self.image_path = path
        self.cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.cv_img = cv2.resize(self.cv_img, (self.width(), self.height()))
        self.image = QImage(self.cv_img.data, self.cv_img.shape[1], self.cv_img.shape[0], self.cv_img.strides[0], QImage.Format_Grayscale8)
        self.setPixmap(QPixmap.fromImage(self.image))

        self.mask = np.zeros((self.height(), self.width()), dtype=np.uint8)

        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        mask_path = os.path.join("dataset", "Mask", f"{name}_mask.png")
        if os.path.exists(mask_path):
            self.load_mask(mask_path)
        else:
            self.update_view()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_at(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.draw_at(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def draw_at(self, pos):
        x, y = pos.x(), pos.y()
        cv2.circle(self.mask, (x, y), self.brush_radius, self.current_class, -1)
        self.update_view()

    def update_view(self):
        if self.image is None:
            return

        base = cv2.cvtColor(self.cv_img.copy(), cv2.COLOR_GRAY2BGR)
        overlay = base.copy()

        red_color = (0, 0, 255)  # OpenCV는 BGR 순서
        mask_region = (self.mask > 0)
        overlay[mask_region] = red_color

        alpha = 0.4  # 투명도 (0.0~1.0)
        blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

        qimg = QImage(blended.data, blended.shape[1], blended.shape[0], blended.strides[0], QImage.Format_BGR888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def save_current_mask(self):
        if self.image_path is None:
            return None
        os.makedirs(os.path.join("dataset", "Mask"), exist_ok=True)
        base = os.path.basename(self.image_path)
        name, _ = os.path.splitext(base)
        path = os.path.join("dataset", "Mask", f"{name}_mask.png")
        cv2.imwrite(path, self.mask)
        return path

    def load_mask(self, path):
        self.mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.resize(self.mask, (self.width(), self.height()), interpolation=cv2.INTER_NEAREST)
        self.update_view()

    def clear_mask(self):
        self.mask[:] = 0
        self.update_view()

    def set_brush_radius(self, radius):
        self.brush_radius = radius
        self.update()
