#main.py
"""
Main entry point for the Defect ROI Labeling Tool.

This script initializes and runs the PyQt5 application, creating and showing the main window.
"""
import sys

from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

