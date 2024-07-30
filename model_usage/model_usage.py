from ultralytics import YOLO
import cv2
import os
import numpy as np
from utils import *
import tensorflow as tf
import torch
import onnxruntime as ort

MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_TYPE = os.getenv('MODEL_TYPE')
MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_OUTPUT = os.getenv('MODEL_INFERENCE_OUTPUT')


def load_model(model_path, model_type):
    if model_type.lower() == 'yolo':
        return YOLO(model_path)
    elif model_type.lower() == 'tensorflow':
        return tf.saved_model.load(model_path)
    elif model_type.lower() == 'pytorch':
        return torch.load(model_path)
    elif model_type.lower() == 'onnx':
        return ort.InferenceSession(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_inference(model, image, model_type):
    if model_type.lower() == 'yolo':
        return model(image)
    elif model_type.lower() == 'tensorflow':
        # Implement TensorFlow inference
        pass
    elif model_type.lower() == 'pytorch':
        # Implement PyTorch inference
        pass
    elif model_type.lower() == 'onnx':
        # Implement ONNX inference
        pass
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def process_results(results, image, model_type):
    if model_type.lower() == 'yolo':
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'{result.names[cls]} {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        pass

    return image


def inference():
    model = load_model(MODEL_PATH, MODEL_TYPE)

    os.makedirs(MODEL_INFERENCE_OUTPUT, exist_ok=True)

    for filename in os.listdir(MODEL_INFERENCE_INPUT):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, filename)
            output_path = os.path.join(MODEL_INFERENCE_OUTPUT, f'detected_{filename}')

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = run_inference(model, image, MODEL_TYPE)

            image = process_results(results, image, MODEL_TYPE)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)

            print(f"Processed: {filename}")

    print("All images processed!")


if __name__ == '__main__':
    check_directory_content(MODEL_PATH + MODEL_INFERENCE_INPUT)
    inference()
