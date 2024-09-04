import os
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2
)
import cv2
from utils import *

MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_FRCNN_OUTPUT_DIR = os.getenv('MODEL_INFERENCE_FRCNN_OUTPUT_DIR')

IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD'))

AVAILABLE_MODELS = {
    'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    'fasterrcnn_mobilenet_v3_large_320_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    'fasterrcnn_resnet50_fpn_v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
}


def load_faster_rcnn_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            return state_dict['model'].eval()

        architecture = state_dict.get('architecture')
        num_classes = state_dict.get('num_classes')

        if not architecture or architecture not in AVAILABLE_MODELS:
            raise ValueError(f"Unsupported or missing architecture: {architecture}")

        model = AVAILABLE_MODELS[architecture](pretrained=False, num_classes=num_classes)
        model.load_state_dict(state_dict['model_state_dict'])
        return model.eval()
    else:
        raise ValueError("Unexpected format of saved model")


def run_inference(model, image):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            model = model.cuda()
        return model(image_tensor)


def process_results(results, image):
    result = results[0]
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    labels = result['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > IOU_THRESHOLD:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f'Class {label}: {score:.2f}'
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


def frcnn_inference():
    models = []
    for filename in os.listdir(MODEL_INFERENCE_FRCNN_OUTPUT_DIR):
        if filename.endswith('.pth'):
            model_path = os.path.join(MODEL_INFERENCE_FRCNN_OUTPUT_DIR, filename)
            try:
                model = load_faster_rcnn_model(model_path)
                models.append((filename, model))
                print(f"Loaded model: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    for image_filename in os.listdir(MODEL_INFERENCE_INPUT):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, image_filename)

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for model_name, model in models:
                try:
                    results = run_inference(model, image)

                    output_image = process_results(results, image.copy())
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

                    model_output_dir = os.path.join(MODEL_INFERENCE_FRCNN_OUTPUT_DIR, os.path.splitext(model_name)[0])
                    os.makedirs(model_output_dir, exist_ok=True)

                    output_filename = (f"{os.path.splitext(image_filename)[0]}_{model_name}"
                                       f"{os.path.splitext(image_filename)[1]}")
                    output_path = os.path.join(model_output_dir, output_filename)

                    cv2.imwrite(output_path, output_image)
                    print(f"Processed {image_filename} with {model_name}. Location: {output_path}")
                except Exception as e:
                    print(f"Error processing {image_filename}: {str(e)}")

    print("All images processed with all models!")


def frcnn_main():
    frcnn_does_exist = check_directory_content([MODEL_INFERENCE_FRCNN_OUTPUT_DIR])

    if frcnn_does_exist:
        frcnn_inference()
    else:
        print("No frcnn Model has been found.")
