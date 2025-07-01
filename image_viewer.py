# image_viewer.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect
import cv2
import numpy as np

class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.rect_start = None
        self.rect_end = None
        self.setScaledContents(True)

    def load_image(self, path):
        self.cv_img = cv2.imread(path)
        self.image = self.cv_to_qt(self.cv_img)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect_start = event.pos()
            self.rect_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.rect_start:
            self.rect_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect_end = event.pos()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rect_start and self.rect_end:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            rect = QRect(self.rect_start, self.rect_end)
            painter.drawRect(rect)

    def has_roi(self):
        return self.rect_start and self.rect_end

    def get_roi_image(self):
        if not self.has_roi():
            return None
        rect = QRect(self.rect_start, self.rect_end).normalized()
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        h_ratio = self.cv_img.shape[0] / self.height()
        w_ratio = self.cv_img.shape[1] / self.width()
        x1, y1, x2, y2 = map(int, [x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio])
        return self.cv_img[y1:y2, x1:x2]

    def cv_to_qt(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        return QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
