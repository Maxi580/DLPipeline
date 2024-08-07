from ultralytics import YOLO
import cv2
import torch
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn,
                                          fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2
from utils import *

MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_OUTPUT = os.getenv('MODEL_INFERENCE_OUTPUT')

IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD'))

YOLO_MODEL_TYPE = 'yolo'
PYTORCH_MODEL_TYPE = 'pytorch'

AVAILABLE_MODELS = {
    'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    'fasterrcnn_mobilenet_v3_large_320_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    'fasterrcnn_resnet50_fpn_v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    'ssd300_vgg16': ssd300_vgg16,
    'retinanet_resnet50_fpn_v2': retinanet_resnet50_fpn_v2,
}


def load_model(model_dir):
    # Dictionary mapping file extensions to model types and loading functions
    model_types = {
        '.pt': (YOLO_MODEL_TYPE, lambda path: YOLO(path)),
        '.pth': (PYTORCH_MODEL_TYPE, load_pytorch_model),
        # '.h5': ('tensorflow', lambda path: tf.keras.models.load_model(path)),
        # '.pb': ('tensorflow', lambda path: tf.saved_model.load(path)),
        # '.onnx': ('onnx', lambda path: onnxruntime.InferenceSession(path)),
    }

    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        _, extension = os.path.splitext(filename)

        if extension in model_types:
            model_type, load_func = model_types[extension]
            try:
                model = load_func(file_path)
                return model_type, model
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

    raise ValueError(f"No supported model file found in {model_dir}")


def get_pytorch_model_architecture(architecture_name, num_classes=None):
    if architecture_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model architecture: {architecture_name}")

    model_constructor = AVAILABLE_MODELS[architecture_name]

    # Check if the model constructor accepts num_classes
    import inspect
    sig = inspect.signature(model_constructor)
    if 'num_classes' in sig.parameters:
        return model_constructor(pretrained=False, num_classes=num_classes if num_classes is not None else 91)
    else:
        return model_constructor(pretrained=False)


def load_pytorch_model(path):
    state_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            return state_dict['model'].eval()

        architecture = state_dict.get('architecture')
        num_classes = state_dict.get('num_classes')

        if not architecture:
            raise ValueError("Architecture information not found in the saved model")

        model = get_pytorch_model_architecture(architecture, num_classes)
        model.load_state_dict(state_dict['model_state_dict'])
        return model.eval()
    else:
        raise ValueError("Unexpected format of saved model")


def run_inference(model, model_type, image):
    if model_type == YOLO_MODEL_TYPE:
        return model(image)
    if model_type == PYTORCH_MODEL_TYPE:
        model.eval()
        with torch.no_grad():
            # Convert image to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                model = model.cuda()
            return model(image_tensor)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def process_results(results, model_type, image):
    if model_type == YOLO_MODEL_TYPE:
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'{result.names[cls]} {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif model_type == PYTORCH_MODEL_TYPE:
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


def inference():
    """
       You need to load every model differently depending on the model type, which is also true for using the model
       In Essence this function uses the first model in /inference/model on every image in /inference/input_images
    """
    try:
        model_type, model = load_model(MODEL_PATH)
    except Exception as e:
        raise ValueError(f"No Model File Found: {e}")

    os.makedirs(MODEL_INFERENCE_OUTPUT, exist_ok=True)
    for filename in os.listdir(MODEL_INFERENCE_INPUT):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, filename)
            output_path = os.path.join(MODEL_INFERENCE_OUTPUT, f'detected_{filename}')

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = run_inference(model, model_type, image)
            image = process_results(results, model_type, image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)

            print(f"Processed: {filename}")

    print("All images processed!")


if __name__ == '__main__':
    does_exist = check_directory_content([MODEL_PATH, MODEL_INFERENCE_INPUT])

    if does_exist:
        inference()
