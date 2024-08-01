import os
import json
from utils import *
from yolo import *
from faster_r_cnn import *

YOLO_WITH_AUGMENTATION = os.getenv('YOLO_WITH_AUGMENTATION')
YOLO_WITHOUT_AUGMENTATION = os.getenv('YOLO_WITHOUT_AUGMENTATION')
FASTER_RCNN_WITH_AUGMENTATION = os.getenv('FASTER_RCNN_WITH_AUGMENTATION')
FASTER_RCNN_WITHOUT_AUGMENTATION = os.getenv('FASTER_RCNN_WITHOUT_AUGMENTATION')

PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')
MODEL_WITH_PIXMIX_AUGMENTATION = os.getenv('MODEL_WITH_PIXMIX_AUGMENTATION')
PIXMIX_OUTPUT_DIR = os.getenv('PIXMIX_OUTPUT_DIR')
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')


def main():
    """Checks if data is available, trains model, saves model and results"""
    create_directory(MODEL_OUTPUT_DIR)
    if YOLO_WITHOUT_AUGMENTATION:
        input_subdirectories = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
        input_image_directories = input_subdirectories[0]
        input_annotations_directories = input_subdirectories[1]
        does_exist = check_directory_content(input_image_directories + input_annotations_directories)
        if does_exist:
            name = 'yolo_without_augmentation'
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(PREPROCESSING_OUTPUT_DIR, data_yaml_path, name)

    if YOLO_WITH_AUGMENTATION:
        input_subdirectories = get_subdirectories(PIXMIX_OUTPUT_DIR)
        input_image_directories = input_subdirectories[0]
        input_annotations_directories = input_subdirectories[1]
        does_exist = check_directory_content(input_image_directories + input_annotations_directories)
        if does_exist:
            name = 'yolo_pixmix'
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(PIXMIX_OUTPUT_DIR, data_yaml_path, name)

    if FASTER_RCNN_WITHOUT_AUGMENTATION:
        input_subdirectories = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
        input_image_directories = input_subdirectories[0]
        input_annotations_directories = input_subdirectories[1]
        does_exist = check_directory_content(input_image_directories + input_annotations_directories)
        if does_exist:
            name = 'faster_rcnn_without_augmentation'
            train_image_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)

    if FASTER_RCNN_WITH_AUGMENTATION:
        input_subdirectories = get_subdirectories(PIXMIX_OUTPUT_DIR)
        input_image_directories = input_subdirectories[0]
        input_annotations_directories = input_subdirectories[1]
        does_exist = check_directory_content(input_image_directories + input_annotations_directories)
        if does_exist:
            name = 'faster_rcnn_with_augmentation'
            train_image_dir = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)


if __name__ == '__main__':
    main()
