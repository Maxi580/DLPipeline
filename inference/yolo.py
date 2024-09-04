from ultralytics import YOLO
import cv2

from utils import *

MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_YOLO_OUTPUT_DIR = os.getenv('MODEL_INFERENCE_YOLO_OUTPUT_DIR')


def load_yolo_models(model_dir):
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pt'):  # Assuming YOLO models have .pt extension
            model_path = os.path.join(model_dir, filename)
            model = YOLO(model_path)
            models.append((filename, model))
    return models


def yolo_inference():
    models = load_yolo_models(MODEL_INFERENCE_YOLO_OUTPUT_DIR)
    if not models:
        print("No YOLO models found in the specified directory.")
        return

    for image_name in os.listdir(MODEL_INFERENCE_INPUT):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            image_path = os.path.join(MODEL_INFERENCE_INPUT, image_name)
            img = cv2.imread(image_path)

            for model_name, model in models:
                results = model(img)

                model_output_dir = os.path.join(MODEL_INFERENCE_YOLO_OUTPUT_DIR, model_name.split('.')[0])
                os.makedirs(model_output_dir, exist_ok=True)

                output_path = os.path.join(model_output_dir, f"result_{image_name}")
                results[0].save(output_path)

                print(f"Processed {image_name} with model {model_name}")


def yolo_main():
    yolo_does_exist = check_directory_content([MODEL_INFERENCE_YOLO_OUTPUT_DIR])

    if yolo_does_exist:
        yolo_inference()
    else:
        print("No YOLO Model has been found.")
