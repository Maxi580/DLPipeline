import yaml
from ultralytics import YOLO
import os

IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT'))
YOLO_MODELS = os.getenv('YOLO_MODELS').split(',')
EPOCHS = int(os.getenv('EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')
MAPPING_FILE = os.getenv('MAPPING_FILE')
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')


def determine_number_of_classes(annotations_path):
    class_ids = set()
    for root, _, files in os.walk(annotations_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
    return len(class_ids)


def create_dataset_yaml(dataset_path, output_path):
    train_images = os.path.join(IMAGES_PATH, TRAIN_PATH)
    train_labels = os.path.join(LABEL_PATH, TRAIN_PATH)
    val_images = os.path.join(IMAGES_PATH, VAL_PATH)
    val_labels = os.path.join(LABEL_PATH, VAL_PATH)

    dataset_dict = {
        'path': dataset_path,
        'train': train_images,
        'val': val_images,
        'train_labels': train_labels,
        'labels': val_labels,
    }

    # Either you specify class names if file exists or just how many classes there are if not
    try:
        with open(MAPPING_FILE, 'r') as mapping_file:
            class_mapping = yaml.safe_load(mapping_file)
        dataset_dict['names'] = {int(class_id): class_name for class_id, class_name in class_mapping.items()}
        print(f"Class mapping loaded from {MAPPING_FILE}")
    except FileNotFoundError:
        print(f"Warning: Mapping file {MAPPING_FILE} not found. Using class IDs only.")
        nc = determine_number_of_classes(train_labels)
        dataset_dict['nc'] = nc
        print(f"Number of classes detected: {nc}")

    with open(output_path, 'w') as f:
        yaml.safe_dump(dataset_dict, f, sort_keys=False)

    print(f"Dataset YAML file created at {output_path}")


def create_yolo_model(input_path, data_yaml_path, name):
    for yolo_model in YOLO_MODELS:
        model = YOLO(yolo_model)

        create_dataset_yaml(input_path, data_yaml_path)

        results = model.train(
            data=data_yaml_path,
            epochs=EPOCHS,
            imgsz=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch=BATCH_SIZE,
            project=MODEL_OUTPUT_DIR,
            name=f"{name}_{yolo_model}",
        )
        print(f"Model {yolo_model} created: {model} {results}")