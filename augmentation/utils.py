import os
import numpy as np
from PIL import Image

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')

WIDTH = os.getenv('WIDTH')
HEIGHT = os.getenv('HEIGHT')

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
        annotation_subdirectories.append(os.path.join(directory, LABEL_PATH, split_sub_directory))
    return image_subdirectories, annotation_subdirectories


def is_directory_empty(directory_path):
    try:
        return len(os.listdir(directory_path)) == 0
    except Exception as e:
        print(f"Could not check contents of directory {directory_path}: {e}")
        return True


def check_directory_content(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
            return False
        elif is_directory_empty(path):
            print(f"Warning: Directory is empty: {path}")
            return False
    return True


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


def read_yolo_annotation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = [[int(float(values[0]))] + list(map(float, values[1:])) for line in lines for values in
                   [line.strip().split()]]
    return annotations


def save_yolo_file(directory, filename, bboxes, class_ids):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_ids):
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
            line = f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
            f.write(line)


def load_image_set(directory_path):
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, filename)
            image_paths.append(image_path)

    return image_paths


