# classifier.py
import onnxruntime
import numpy as np
import cv2

CLASS_NAMES = ["Scratch", "Dust", "Dent"]

class DefectClassifier:
    def __init__(self, model_path="model/defect_classifier.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image):
        # 전처리: Grayscale, Resize, Normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        input_tensor = image.reshape(1, 1, 224, 224).astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        probs = outputs[0][0]  # Softmax 출력
        print("Softmax probs:", probs)
        pred_idx = int(np.argmax(probs))
        return CLASS_NAMES[pred_idx]
