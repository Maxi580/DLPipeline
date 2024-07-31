from ultralytics import YOLO
import cv2
from utils import *

MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_TYPE = os.getenv('MODEL_TYPE')
MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_OUTPUT = os.getenv('MODEL_INFERENCE_OUTPUT')


def load_yolo_model(model_dir):
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in '{model_dir}'")
    else:
        model_path = os.path.join(model_dir, pt_files[0])
        return YOLO(model_path)


def run_inference(model, image):
    if MODEL_TYPE.lower() == 'yolo':
        return model(image)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def process_results(results, image):
    if MODEL_TYPE.lower() == 'yolo':
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
    if MODEL_TYPE.lower() == 'yolo':
        model = load_yolo_model(MODEL_PATH)
    else:
        raise ValueError(f"{MODEL_TYPE} Model could not be found in {MODEL_PATH}")

    os.makedirs(MODEL_INFERENCE_OUTPUT, exist_ok=True)
    for filename in os.listdir(MODEL_INFERENCE_INPUT):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, filename)
            output_path = os.path.join(MODEL_INFERENCE_OUTPUT, f'detected_{filename}')

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = run_inference(model, image)

            image = process_results(results, image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)

            print(f"Processed: {filename}")

    print("All images processed!")


if __name__ == '__main__':
    check_directory_content(MODEL_PATH + MODEL_INFERENCE_INPUT)
    inference()
