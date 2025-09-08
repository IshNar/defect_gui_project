# brush_image_viewer.py

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2
import os

class BrushImageViewer(QLabel):
    """A custom QLabel widget for displaying images and drawing masks.

    This widget allows users to load an image, draw on it with a brush of
    adjustable size and class, and save the resulting mask. It handles mouse
    events for drawing and updates the view to show the base image with a
    semi-transparent mask overlay.
    """
    def __init__(self, parent=None):
        """Initializes the BrushImageViewer.

        Args:
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setFixedSize(512, 512)
        self.setStyleSheet("border: 1px solid black")
        self.image = None
        self.mask = None
        self.image_path = None
        self.drawing = False
        self.brush_radius = 8
        self.current_class = 1

    def set_class(self, class_id):
        """Sets the current class for drawing.

        Args:
            class_id: The integer ID of the class.
        """
        self.current_class = class_id

    def load_image(self, path):
        """Loads an image from the given path.

        Args:
            path: The file path of the image to load.
        """
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
        """Handles mouse press events to start drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_at(event.pos())

    def mouseMoveEvent(self, event):
        """Handles mouse move events to draw on the mask."""
        if self.drawing:
            self.draw_at(event.pos())

    def mouseReleaseEvent(self, event):
        """Handles mouse release events to stop drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def draw_at(self, pos):
        """Draws a circle on the mask at the given position."""
        x, y = pos.x(), pos.y()
        cv2.circle(self.mask, (x, y), self.brush_radius, self.current_class, -1)
        self.update_view()

    def update_view(self):
        """Updates the display with the image and mask overlay."""
        if self.image is None:
            return

        base = cv2.cvtColor(self.cv_img.copy(), cv2.COLOR_GRAY2BGR)
        overlay = base.copy()

        red_color = (0, 0, 255)  # Red in BGR
        mask_region = (self.mask > 0)
        overlay[mask_region] = red_color

        alpha = 0.4  # Transparency
        blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

        qimg = QImage(blended.data, blended.shape[1], blended.shape[0], blended.strides[0], QImage.Format_BGR888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def save_current_mask(self):
        """Saves the current mask to a file.

        Returns:
            The path where the mask was saved, or None if no image is loaded.
        """
        if self.image_path is None:
            return None
        os.makedirs(os.path.join("dataset", "Mask"), exist_ok=True)
        base = os.path.basename(self.image_path)
        name, _ = os.path.splitext(base)
        path = os.path.join("dataset", "Mask", f"{name}_mask.png")
        cv2.imwrite(path, self.mask)
        return path

    def load_mask(self, path):
        """Loads a mask from the given path.

        Args:
            path: The file path of the mask to load.
        """
        self.mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.resize(self.mask, (self.width(), self.height()), interpolation=cv2.INTER_NEAREST)
        self.update_view()

    def clear_mask(self):
        """Clears the current mask."""
        self.mask[:] = 0
        self.update_view()

    def set_brush_radius(self, radius):
        """Sets the brush radius.

        Args:
            radius: The new radius for the brush.
        """
        self.brush_radius = radius
        self.update()
