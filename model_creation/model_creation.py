from utils import *
from yolo import *
from faster_r_cnn import *
from unet import create_u_net_model

LABEL_PATH = os.getenv('LABEL_PATH')
MASK_PATH = os.getenv('MASK_PATH')

YOLO_WITH_AUGMENTATION = os.getenv('YOLO_WITH_AUGMENTATION')
YOLO_WITHOUT_AUGMENTATION = os.getenv('YOLO_WITHOUT_AUGMENTATION')
FASTER_RCNN_WITH_AUGMENTATION = os.getenv('FASTER_RCNN_WITH_AUGMENTATION')
FASTER_RCNN_WITHOUT_AUGMENTATION = os.getenv('FASTER_RCNN_WITHOUT_AUGMENTATION')
UNET_WITH_AUGMENTATION = os.getenv('UNET_WITH_AUGMENTATION')
UNET_WITHOUT_AUGMENTATION = os.getenv('UNET_WITHOUT_AUGMENTATION')

DETECTION_PREPROCESSING_OUTPUT_DIR = os.getenv('DETECTION_PREPROCESSING_OUTPUT_DIR')
SEGMENTATION_PREPROCESSING_OUTPUT_DIR = os.getenv('SEGMENTATION_PREPROCESSING_OUTPUT_DIR')
DETECTION_AUGMENTATION_OUTPUT_DIR = os.getenv('DETECTION_AUGMENTATION_OUTPUT_DIR')
SEGMENTATION_AUGMENTATION_OUTPUT_DIR = os.getenv('SEGMENTATION_AUGMENTATION_OUTPUT_DIR')

MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')


def main():
    """Checks if data is available, trains selected model(s), saves model and results
       I am well aware that == 'TRUE' is bad but bool(TRUE) == bool(FALSE) == TRUE (Workarounds dont improve QOC
       from my point of view."""
    create_directory(MODEL_OUTPUT_DIR)

    detection_preprocessing_input_subdirectories = get_subdirectories(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH)
    detection_preprocessing_input_image_directories = detection_preprocessing_input_subdirectories[0]
    detection_preprocessing_input_annotation_directories = detection_preprocessing_input_subdirectories[1]
    detection_preprocessing_does_exist = check_directory_content(detection_preprocessing_input_image_directories +
                                                                 detection_preprocessing_input_annotation_directories)

    detection_augmentation_input_subdirectories = get_subdirectories(DETECTION_AUGMENTATION_OUTPUT_DIR, LABEL_PATH)
    detection_augmentation_input_image_directories = detection_augmentation_input_subdirectories[0]
    detection_augmentation_input_annotation_directories = detection_augmentation_input_subdirectories[1]
    detection_augmentation_does_exist = check_directory_content(detection_augmentation_input_image_directories +
                                                                detection_augmentation_input_annotation_directories)

    segmentation_preprocessing_input_subdirectories = get_subdirectories(SEGMENTATION_PREPROCESSING_OUTPUT_DIR,
                                                                         MASK_PATH)
    segmentation_preprocessing_input_image_directories = segmentation_preprocessing_input_subdirectories[0]
    segmentation_preprocessing_input_annotation_directories = segmentation_preprocessing_input_subdirectories[1]
    segmentation_preprocessing_does_exist = check_directory_content(segmentation_preprocessing_input_image_directories +
                                                                    segmentation_preprocessing_input_annotation_directories)

    segmentation_augmentation_input_subdirectories = get_subdirectories(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, MASK_PATH)
    segmentation_augmentation_input_image_directories = segmentation_augmentation_input_subdirectories[0]
    segmentation_augmentation_input_annotation_directories = segmentation_augmentation_input_subdirectories[1]
    segmentation_augmentation_does_exist = check_directory_content(segmentation_augmentation_input_image_directories +
                                                                   segmentation_augmentation_input_annotation_directories)

    if detection_preprocessing_does_exist:
        if YOLO_WITHOUT_AUGMENTATION == 'TRUE':
            name = 'yolo_no_aug'
            print(f"Starting with: {name}")
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(DETECTION_PREPROCESSING_OUTPUT_DIR, data_yaml_path, name)
        if FASTER_RCNN_WITHOUT_AUGMENTATION == 'TRUE':
            name = 'faster_rcnn_no_aug'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)

    if detection_augmentation_does_exist:
        if YOLO_WITH_AUGMENTATION == 'TRUE':
            name = 'yolo_aug'
            print(f"Starting with: {name}")
            data_yaml_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.yaml")
            create_yolo_model(DETECTION_AUGMENTATION_OUTPUT_DIR, data_yaml_path, name)
        if FASTER_RCNN_WITH_AUGMENTATION == 'TRUE':
            name = 'faster_rcnn_aug'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)

    if segmentation_augmentation_does_exist:
        print("segmentation_augmentation_does_exist")
        if UNET_WITH_AUGMENTATION == 'TRUE':
            name = 'UNET_aug'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, MASK_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, MASK_PATH, VAL_PATH)
            create_u_net_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)

    if segmentation_preprocessing_does_exist:
        if UNET_WITHOUT_AUGMENTATION == 'TRUE':
            name = 'UNET_no_aug'
            print(f"Starting with: {name}")
            train_image_dir = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            train_label_dir = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH, TRAIN_PATH)
            val_image_dir = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            val_label_dir = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH, VAL_PATH)
            create_u_net_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name)


if __name__ == '__main__':
    main()
