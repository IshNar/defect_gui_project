# log_writer.py
"""Provides a simple logging function for the application."""

def log(log_widget, message):
    """Logs a message to a QTextEdit widget and the console.

    Args:
        log_widget: The QTextEdit widget to append the message to.
        message: The message string to log.
    """
    log_widget.append(message)
    log_widget.verticalScrollBar().setValue(log_widget.verticalScrollBar().maximum())
    print(message)
