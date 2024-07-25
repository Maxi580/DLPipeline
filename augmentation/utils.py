import os
import numpy as np

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')

DATA_SPLIT_DIRECTORY = [TRAIN_PATH, VAL_PATH]


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def get_subdirectories(directory):
    """Appends val and train to and returns these paths"""
    image_subdirectories = []
    annotation_subdirectories = []
    for split_sub_directory in DATA_SPLIT_DIRECTORY:
        image_subdirectories.append(os.path.join(directory, IMAGES_PATH, split_sub_directory))
        annotation_subdirectories.append(os.path.join(directory, ANNOTATION_PATH, split_sub_directory))
    return image_subdirectories, annotation_subdirectories


def is_directory_empty(directory_path):
    try:
        return len(os.listdir(directory_path)) == 0
    except Exception as e:
        print("Error")
        print(f"Could not check contents of directory {directory_path}: {e}")
        return True


def check_directory_content(paths):
    print(f"PATHS: {paths}")
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
        elif is_directory_empty(path):
            print(f"Warning: Directory is empty: {path}")


def read_yolo_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    bboxes = []
    class_ids = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # Ensure we have all 5 expected values
            class_id, x_center, y_center, width, height = map(float, parts)

            # Validate the bounding box coordinates
            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                bboxes.append([x_center, y_center, width, height])
                class_ids.append(int(class_id))
            else:
                print(f"Warning: Invalid bounding box in {file_path}: {line.strip()}")
        else:
            print(f"Warning: Skipping invalid line in {file_path}: {line.strip()}")

    return np.array(bboxes), np.array(class_ids)


def save_yolo_file(directory, filename, bboxes, class_ids):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_ids):
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
            line = f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
            f.write(line)


def yolo_to_albumentations(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return [max(0, min(1, x_min)), max(0, min(1, y_min)),
            max(0, min(1, x_max)), max(0, min(1, y_max))]


def albumentations_to_yolo(bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]
