# Defect ROI Labeling Tool

This project provides a graphical user interface (GUI) application for labeling regions of interest (ROIs) in images, specifically for identifying surface defects like scratches, dents, and dust. The tool allows users to draw masks on images, save the corresponding ROIs, and train a classifier to automatically identify these defects.

## Features

- **Image Loading**: Load images in various formats (PNG, JPG, BMP).
- **ROI Labeling**: Draw masks on images using an adjustable brush to label defect areas.
- **Class Selection**: Assign a class label (e.g., "Scratch", "Dent", "Dust") to each ROI.
- **Mask Saving**: Save the drawn masks for later use.
- **ROI Dataset Creation**: Automatically save the labeled ROIs to a structured dataset directory.
- **Classifier Training**: Train a ResNet-based image classifier on the created ROI dataset.
- **Prediction**: Use the trained classifier to predict the class of a new ROI.
- **Evaluation**: Evaluate the classifier's performance with a classification report and confusion matrix.

## Project Structure

The project is organized into several Python scripts:

- `main.py`: The main entry point for the application.
- `mainwindow.py`: Defines the main application window and its UI components.
- `brush_image_viewer.py`: A custom widget for displaying and drawing on images.
- `dataset_saver.py`: A utility for saving labeled ROIs to the dataset.
- `log_writer.py`: A simple logging utility.
- `train_roi_classifier.py`: Contains the model definition and training loop for the ROI classifier.
- `predict_roi_class.py`: Defines the `ROIClassifier` class for making predictions.
- `evaluate_roi_classifier.py`: A script for evaluating the trained classifier.
- `roi_classifier_dataset.py`: Defines the PyTorch `Dataset` for loading the ROI data.
- `create_model.py`: A script to create, train, and export a simple CNN model.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Load an image:**
   - Click the "Load Image" button and select an image file.

3. **Label a defect:**
   - Select a class from the dropdown menu (e.g., "Scratch").
   - Adjust the brush size using the slider.
   - Click and drag the mouse on the image to draw a mask over the defect.

4. **Save the mask:**
   - Click "Save Mask" to save the hand-drawn mask. The mask will be saved in the `dataset/Mask` directory.

5. **Train the classifier:**
   - Click the "Train ROI Classifier" button to start the training process. The progress will be displayed in the log view. The trained model will be saved as `roi_classifier.pth` and `roi_classifier.onnx`.

6. **Predict a defect:**
   - Load an image for which you have already saved a mask.
   - Click the "Predict ROI Class" button to see the predicted class.

7. **Evaluate the classifier:**
   - Click the "Evaluate Classifier" button to generate a classification report and a confusion matrix. The confusion matrix will be saved as `confusion_matrix.png`.
