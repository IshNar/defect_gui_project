# mainwindow.py (updated with evaluation)
"""
Main window for the Defect ROI Labeling Tool.

This script defines the main window of the application, which includes
the user interface for loading images, drawing masks, training a classifier,
and predicting defect classes.
"""
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
    """Main window for the Defect ROI Labeling Tool.

    This class sets up the user interface and connects signals and slots for
    the application's functionality, including loading images, labeling ROIs,
    training a classifier, and predicting defect classes.
    """
    def __init__(self):
        """Initializes the main window and all its widgets."""
        super().__init__()
        self.loaded_image_path = None

        self.setWindowTitle("Defect ROI Labeling Tool")
        self.resize(1000, 800)

        # Left panel: List of ROIs
        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self.display_selected_roi)

        # Right panel: Image viewer
        self.preview_label = BrushImageViewer(self)
        self.preview_label.setFixedSize(448, 448)
        self.preview_label.setStyleSheet("border: 1px solid black")

        # Main layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.roi_list, 1)
        hbox.addWidget(self.preview_label, 1)

        # Buttons
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

        # Class selector
        self.class_selector = QComboBox()
        self.class_selector.addItems(["Scratch", "Dent", "Dust"])
        self.class_selector.currentIndexChanged.connect(self.update_brush_class)

        # Prediction label
        self.predicted_label = QLabel("Prediction: -")
        self.predicted_label.setStyleSheet("font-weight: bold; font-size: 14px")

        # Brush size slider
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(self.preview_label.brush_radius)
        self.brush_slider.valueChanged.connect(self.update_brush_radius)

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # Vertical layout for controls
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

        # Main container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect signals to slots
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_roi)

        # Initial population of the ROI list
        self.update_roi_list()
        log(self.log_view, "UI Initialized")

    def update_brush_class(self, index):
        """
        Updates the brush class based on the class selector's index.

        Args:
            index: The new index of the class selector.
        """
        class_id = index + 1
        self.preview_label.set_class(class_id)
        log(self.log_view, f"🖌️ Brush class set to {self.class_selector.currentText()} ({class_id})")

    def update_brush_radius(self, value):
        """
        Updates the brush radius based on the slider's value.

        Args:
            value: The new value of the brush size slider.
        """
        self.preview_label.set_brush_radius(value)
        log(self.log_view, f"🖌️ Brush size set to {value}")

    def save_mask_manual(self):
        """Saves the current mask to a file."""
        path = self.preview_label.save_current_mask()
        if path:
            log(self.log_view, f"💾 Mask saved: {path}")

    def load_image(self):
        """Opens a file dialog to load an image for labeling."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.loaded_image_path = path
            self.preview_label.load_image(path)
            log(self.log_view, f"Image loaded: {path}")

    def save_roi(self):
        """Saves the currently selected ROI to the dataset.

        This method is currently not fully implemented, as the `BrushImageViewer`
        class does not have `has_roi` or `get_roi_image` methods.
        """
        if hasattr(self.preview_label, 'has_roi') and self.preview_label.has_roi():
            roi = self.preview_label.get_roi_image()
            label = self.class_selector.currentText()
            save_roi(roi, label)
            log(self.log_view, f"ROI saved to class: {label}")
        else:
            log(self.log_view, "No ROI selected.")

    def display_selected_roi(self, item):
        """
        Displays the selected ROI from the list.

        Args:
            item: The QListWidgetItem that was clicked.
        """
        if self.preview_label.mask is not None and self.preview_label.image_path is not None:
            path_saved = self.preview_label.save_current_mask()
            if path_saved:
                log(self.log_view, f"💾 Mask auto-saved: {path_saved}")
        path = item.text()
        self.preview_label.load_image(path)
        self.predicted_label.setText("Prediction: -")
        log(self.log_view, f"Selected: {path}")

    def run_training(self):
        """Runs the ROI classifier training in a background thread."""
        def background_train():
            run_train_from_ui(lambda msg: log(self.log_view, msg))
        thread = threading.Thread(target=background_train)
        thread.start()

    def run_evaluation(self):
        """Runs the ROI classifier evaluation in a background thread."""
        def background_eval():
            evaluate_roi_classifier(log_fn=lambda msg: log(self.log_view, msg))
        thread = threading.Thread(target=background_eval)
        thread.start()

    def predict_roi_class(self):
        """Predicts the class of the current ROI using the trained classifier."""
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
            classifier = ROIClassifier()
            result = classifier.predict(image_path, mask_path)
            self.predicted_label.setText(f"Prediction: {result}")
            log(self.log_view, f"🧩 Predicted class: {result}")
        except FileNotFoundError:
            log(self.log_view, "❌ Model not found. Please train first.")


    def update_roi_list(self):
        """
        Updates the list of ROIs by scanning the dataset directory.

        This method clears the current list and repopulates it by scanning
        the subdirectories of the `dataset` folder.
        """
        self.roi_list.clear()
        dataset_root = "dataset"
        for class_name in os.listdir(dataset_root):
            if class_name == "Mask":
                continue  # Skip the Mask folder
            class_path = os.path.join(dataset_root, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".png", ".jpg", ".bmp")):
                        full_path = os.path.join(class_path, fname)
                        self.roi_list.addItem(full_path)
