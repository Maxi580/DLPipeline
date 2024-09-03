from pixmix import *
import shutil

DETECTION_PREPROCESSING_OUTPUT_DIR = os.getenv('DETECTION_PREPROCESSING_OUTPUT_DIR')
SEGMENTATION_PREPROCESSING_OUTPUT_DIR = os.getenv('SEGMENTATION_PREPROCESSING_OUTPUT_DIR')
DETECTION_AUGMENTATION_OUTPUT_DIR = os.getenv('DETECTION_AUGMENTATION_OUTPUT_DIR')
SEGMENTATION_AUGMENTATION_OUTPUT_DIR = os.getenv('SEGMENTATION_AUGMENTATION_OUTPUT_DIR')

YOLO_WITH_AUGMENTATION = bool(os.getenv('YOLO_WITH_AUGMENTATION'))
FASTER_RCNN_WITH_AUGMENTATION = bool(os.getenv('YOLO_WITH_AUGMENTATION'))
UNET_WITH_AUGMENTATION = bool(os.getenv('UNET_WITH_AUGMENTATION'))

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
MASK_PATH = os.getenv('MASK_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')


def copy_val_data(input_image_dir_val, output_image_dir_val, input_dir_label_val, output_dir_label_val):
    # Copy Val Data
    for img in os.listdir(input_image_dir_val):
        shutil.copy2(os.path.join(input_image_dir_val, img), os.path.join(output_image_dir_val, img))

    for ann in os.listdir(input_dir_label_val):
        shutil.copy2(os.path.join(input_dir_label_val, ann),
                     os.path.join(output_dir_label_val, ann))


def main():
    """Essentially checks if preprocessed input data is available and creates output directories
       Then if the user wants to train Models with augmentation, the augmentation proceeds.
       It is to note that training data gets augmented but val data not. (Easily changeable if needed)"""
    try:
        # Check if Detection Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH)
        object_detection_does_exist = check_directory_content(input_image_paths + input_annotation_paths)

        # Check if Segmentation Training Data is availiable
        input_image_paths, input_mask_paths = get_subdirectories(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH)
        object_segmentation_does_exist = check_directory_content(input_image_paths + input_mask_paths)

        if object_detection_does_exist:
            input_image_dir_train = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            input_annotation_dir_train = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            output_image_dir_train = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            output_annotation_dir_train = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)

            input_image_dir_val = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            input_annotation_dir_val = os.path.join(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            output_image_dir_val = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            output_annotation_dir_val = os.path.join(DETECTION_AUGMENTATION_OUTPUT_DIR, LABEL_PATH, VAL_PATH)

            if YOLO_WITH_AUGMENTATION or FASTER_RCNN_WITH_AUGMENTATION:
                # Create Output Directories
                output_image_paths, output_annotation_paths = get_subdirectories(DETECTION_AUGMENTATION_OUTPUT_DIR,
                                                                                 LABEL_PATH)
                create_directories(output_image_paths + output_annotation_paths)

                # Augment Training Data
                detection_pixmix(input_image_dir_train, output_image_dir_train, input_annotation_dir_train,
                                 output_annotation_dir_train)

                copy_val_data(input_image_dir_val, output_image_dir_val, input_annotation_dir_val,
                              output_annotation_dir_val)

        if object_segmentation_does_exist:
            input_image_dir_train = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            input_mask_dir_train = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH, TRAIN_PATH)
            output_image_dir_train = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            output_mask_dir_train = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, MASK_PATH, TRAIN_PATH)

            input_image_dir_val = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            input_mask_dir_val = os.path.join(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH, VAL_PATH)
            output_image_dir_val = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            output_mask_dir_val = os.path.join(SEGMENTATION_AUGMENTATION_OUTPUT_DIR, MASK_PATH, VAL_PATH)

            if UNET_WITH_AUGMENTATION:
                # Create Output Directories
                output_image_paths, output_mask_paths = get_subdirectories(SEGMENTATION_AUGMENTATION_OUTPUT_DIR,
                                                                           MASK_PATH)
                create_directories(output_image_paths + output_mask_paths)

                # Augment Training Data
                segmentation_pixmix(input_image_dir_train, output_image_dir_train, input_mask_dir_train,
                                    output_mask_dir_train)

                copy_val_data(input_image_dir_val, output_image_dir_val, input_mask_dir_val, output_mask_dir_val)

    except ValueError as e:
        print(f"Error occurred during augmentation: {e}")


if __name__ == '__main__':
    main()
