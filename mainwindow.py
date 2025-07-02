# mainwindow.py (updated with evaluation)

from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QVBoxLayout, QWidget, QSlider, QListWidget, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from brush_image_viewer import BrushImageViewer
from dataset_saver import save_roi
from log_writer import log
import os
import threading
from train_roi_classifier import run_train_from_ui
from predict_roi_class import ROIClassifier
from evaluate_roi_classifier import evaluate_roi_classifier

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.classifier = ROIClassifier()
        self.loaded_image_path = None

        self.setWindowTitle("Defect ROI Labeling Tool")
        self.resize(800, 600)

        self.roi_list = QListWidget()
        self.preview_label = BrushImageViewer(self)
        self.preview_label.setFixedSize(448, 448)
        self.preview_label.setStyleSheet("border: 1px solid black")
        self.roi_list.itemClicked.connect(self.display_selected_roi)

        hbox = QHBoxLayout()
        hbox.addWidget(self.roi_list, 1)
        hbox.addWidget(self.preview_label, 1)

        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save ROI")
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask_manual)
        self.train_button = QPushButton("Train ROI Classifier")
        self.train_button.clicked.connect(self.run_training)
        self.predict_button = QPushButton("Predict ROI Class")
        self.predict_button.clicked.connect(self.predict_roi_class)
        self.eval_button = QPushButton("Evaluate Classifier")
        self.eval_button.clicked.connect(self.run_evaluation)

        self.class_selector = QComboBox()
        self.class_selector.addItems(["Scratch", "Dent", "Dust"])
        self.class_selector.currentIndexChanged.connect(self.update_brush_class)

        self.predicted_label = QLabel("Prediction: -")
        self.predicted_label.setStyleSheet("font-weight: bold; font-size: 14px")

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(self.preview_label.brush_radius)
        self.brush_slider.valueChanged.connect(self.update_brush_radius)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        layout.addWidget(QLabel("Brush Size"))
        layout.addWidget(self.brush_slider)
        layout.addWidget(self.class_selector)
        layout.addWidget(self.save_mask_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.predicted_label)
        layout.addWidget(self.log_view)
        layout.addWidget(self.train_button)
        layout.addWidget(self.eval_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_roi)

        self.update_roi_list()
        log(self.log_view, "UI Initialized")

    def update_brush_class(self, index):
        class_id = index + 1
        self.preview_label.set_class(class_id)
        log(self.log_view, f"🖌️ Brush class set to {self.class_selector.currentText()} ({class_id})")

    def update_brush_radius(self, value):
        self.preview_label.set_brush_radius(value)
        log(self.log_view, f"🖌️ Brush size set to {value}")

    def save_mask_manual(self):
        path = self.preview_label.save_current_mask()
        if path:
            log(self.log_view, f"💾 Mask saved: {path}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.loaded_image_path = path
            self.preview_label.load_image(path)
            log(self.log_view, f"Image loaded: {path}")

    def save_roi(self):
        if self.preview_label.has_roi():
            roi = self.preview_label.get_roi_image()
            label = self.class_selector.currentText()
            save_roi(roi, label)
            log(self.log_view, f"ROI saved to class: {label}")
        else:
            log(self.log_view, "No ROI selected.")

    def display_selected_roi(self, item):
        if self.preview_label.mask is not None and self.preview_label.image_path is not None:
            path_saved = self.preview_label.save_current_mask()
            if path_saved:
                log(self.log_view, f"💾 Mask auto-saved: {path_saved}")
        path = item.text()
        self.preview_label.load_image(path)
        self.predicted_label.setText("Prediction: -")
        log(self.log_view, f"Selected: {path}")

    def run_training(self):
        def background_train():
            run_train_from_ui(lambda msg: log(self.log_view, msg))
        thread = threading.Thread(target=background_train)
        thread.start()

    def run_evaluation(self):
        def background_eval():
            evaluate_roi_classifier(log_fn=lambda msg: log(self.log_view, msg))
        thread = threading.Thread(target=background_eval)
        thread.start()

    def predict_roi_class(self):
        if not self.preview_label.image_path:
            log(self.log_view, "⚠️ No image loaded for prediction.")
            return

        image_path = self.preview_label.image_path
        base = os.path.basename(image_path)
        name, _ = os.path.splitext(base)
      # Masks are stored in a single folder under the dataset
        mask_path = os.path.join("dataset", "Mask", f"{name}_mask.png")

        if not os.path.exists(mask_path):
            log(self.log_view, f"❌ Mask not found: {mask_path}")
            return

        try:
            classifier = ROIClassifier()  # ✅ 이 시점에만 생성
            result = classifier.predict(image_path, mask_path)
            self.predicted_label.setText(f"Prediction: {result}")
            log(self.log_view, f"🧩 Predicted class: {result}")
        except FileNotFoundError:
            log(self.log_view, "❌ Model not found. Please train first.")


    def update_roi_list(self):
        self.roi_list.clear()
        dataset_root = "dataset"
        for class_name in os.listdir(dataset_root):
            if class_name == "Mask":
                continue  # Mask 폴더는 제외
            class_path = os.path.join(dataset_root, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".png", ".jpg", ".bmp")):
                        full_path = os.path.join(class_path, fname)
                        self.roi_list.addItem(full_path)
