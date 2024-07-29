import os
import json
from utils import *
from yolo import *

CREATE_YOLO_MODEL = os.getenv('CREATE_YOLO_MODEL')

WITHOUT_AUGMENTATION = os.getenv('WITHOUT_AUGMENTATION')
PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')
PIXMIX_AUGMENTATION = os.getenv('PIXMIX_AUGMENTATION')
PIXMIX_OUTPUT_DIR = os.getenv('PIXMIX_OUTPUT_DIR')
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')


def main():
    """Checks if data is available, trains model, saves model and results"""
    create_directory(MODEL_OUTPUT_DIR)
    if CREATE_YOLO_MODEL:
        if WITHOUT_AUGMENTATION:
            input_subdirectories = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
            input_image_directories = input_subdirectories[0]
            input_annotations_directories = input_subdirectories[1]
            does_exist = check_directory_content(input_image_directories + input_annotations_directories)
            if does_exist:
                name = 'yolo_without_augmentation'
                data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
                create_yolo_model(PREPROCESSING_OUTPUT_DIR, data_yaml_path, name)

        if PIXMIX_AUGMENTATION:
            input_subdirectories = get_subdirectories(PIXMIX_OUTPUT_DIR)
            input_image_directories = input_subdirectories[0]
            input_annotations_directories = input_subdirectories[1]
            does_exist = check_directory_content(input_image_directories + input_annotations_directories)
            if does_exist:
                name = 'yolo_pixmix'
                data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
                create_yolo_model(PIXMIX_OUTPUT_DIR, data_yaml_path, name)


if __name__ == '__main__':
    main()
