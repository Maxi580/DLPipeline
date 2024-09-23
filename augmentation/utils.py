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


def check_directory_content(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
            return False
        elif not os.listdir(path):
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


def yolo_to_coco(annotations, image_width, image_height):
    """Converts annotations from yolo into coco format, because we have albumentation issues
       where bbox values after aug are not between 0.0 and 1.0 but e.g. 1.0 * e^-7 which causes an error
    """
    bboxes = []
    class_labels = []
    for ann in annotations:
        class_id, x_center, y_center, bbox_width, bbox_height = ann
        x_min = int((x_center - bbox_width / 2) * image_width)
        y_min = int((y_center - bbox_height / 2) * image_height)
        bbox_width = int(bbox_width * image_width)
        bbox_height = int(bbox_height * image_height)
        bboxes.append([x_min, y_min, bbox_width, bbox_height])
        class_labels.append(int(class_id))
    return bboxes, class_labels


def coco_to_yolo(bboxes, labels, image_width, image_height):
    """Converts annotations back from coco format to yolo format"""
    yolo_annotations = []
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, bbox_width, bbox_height = bbox
        x_center = (x_min + bbox_width / 2) / image_width
        y_center = (y_min + bbox_height / 2) / image_height
        width = bbox_width / image_width
        height = bbox_height / image_height
        yolo_annotations.append([int(label), x_center, y_center, width, height])
    return yolo_annotations
