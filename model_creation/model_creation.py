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
    """Checks if data is available, trains selected model(s), saves model and results
       I am well aware that == 'TRUE' is bad but bool(TRUE) == bool(FALSE) == TRUE (Workarounds dont improve QOC
       from my point of view."""
    create_directory(MODEL_OUTPUT_DIR)

    print(f"YOLO_WITHOUT_AUGMENTATION = {YOLO_WITHOUT_AUGMENTATION}")
    print(f"FASTER_RCNN_WITHOUT_AUGMENTATION = {FASTER_RCNN_WITHOUT_AUGMENTATION}")
    print(f"YOLO_WITH_AUGMENTATION = {YOLO_WITH_AUGMENTATION}")
    print(f"FASTER_RCNN_WITH_AUGMENTATION = {FASTER_RCNN_WITH_AUGMENTATION}")

    input_subdirectories = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
    input_image_directories = input_subdirectories[0]
    input_annotations_directories = input_subdirectories[1]
    preprocessing_does_exist = check_directory_content(input_image_directories + input_annotations_directories)

    print(preprocessing_does_exist)
    if preprocessing_does_exist:
        if YOLO_WITHOUT_AUGMENTATION == 'TRUE':
            name = 'yolo_without_augmentation'
            print(f"Starting with: {name}")
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(PREPROCESSING_OUTPUT_DIR, data_yaml_path, name)
        if FASTER_RCNN_WITHOUT_AUGMENTATION == 'TRUE':
            name = 'faster_rcnn_with_augmentation'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)

    input_subdirectories = get_subdirectories(PIXMIX_OUTPUT_DIR)
    input_image_directories = input_subdirectories[0]
    input_annotations_directories = input_subdirectories[1]
    augmentation_does_exist = check_directory_content(input_image_directories + input_annotations_directories)
    if augmentation_does_exist:
        if YOLO_WITH_AUGMENTATION == 'TRUE':
            name = 'yolo_pixmix'
            print(f"Starting with: {name}")
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(PIXMIX_OUTPUT_DIR, data_yaml_path, name)
        if FASTER_RCNN_WITH_AUGMENTATION == 'TRUE':
            name = 'faster_rcnn_with_augmentation'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)


if __name__ == '__main__':
    main()
