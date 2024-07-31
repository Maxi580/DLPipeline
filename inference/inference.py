from ultralytics import YOLO
import cv2
from utils import *

MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_OUTPUT = os.getenv('MODEL_INFERENCE_OUTPUT')

YOLO_MODEL_TYPE = 'yolo'


def load_model(model_dir):
    # Dictionary mapping file extensions to model types and loading functions
    model_types = {
        '.pt': (YOLO_MODEL_TYPE, lambda path: YOLO(path)),
        # '.pth': ('pytorch', lambda path: torch.load(path)),
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


def run_inference(model, model_type, image):
    if model_type == YOLO_MODEL_TYPE:
        return model(image)
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
    else:
        pass

    return image


def inference():
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
    check_directory_content(MODEL_PATH + MODEL_INFERENCE_INPUT)
    inference()
