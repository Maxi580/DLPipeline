import os
import numpy as np
from PIL import Image

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')

WIDTH = os.getenv('WIDTH')
HEIGHT = os.getenv('HEIGHT')

DATA_SPLIT_DIRECTORY = [TRAIN_PATH, VAL_PATH]


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def get_subdirectories(directory, second_subdirectory):
    """Appends val and train to and returns these paths"""
    image_subdirectories = []
    annotation_subdirectories = []
    for split_sub_directory in DATA_SPLIT_DIRECTORY:
        image_subdirectories.append(os.path.join(directory, IMAGES_PATH, split_sub_directory))
        annotation_subdirectories.append(os.path.join(directory, second_subdirectory, split_sub_directory))
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


def read_yolo_annotation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = [[int(float(values[0]))] + list(map(float, values[1:])) for line in lines for values in
                   [line.strip().split()]]
    return annotations


def load_image_set(directory_path):
    """Designed to load every Fractal, so that you can randomly choose 1"""
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, filename)
            image_paths.append(image_path)

    return image_paths
