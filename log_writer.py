# log_writer.py
def log(log_widget, message):
    log_widget.append(message)
    log_widget.verticalScrollBar().setValue(log_widget.verticalScrollBar().maximum())
    print(message)
