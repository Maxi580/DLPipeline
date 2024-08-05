from pixmix import *
import shutil

PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')
PIXMIX_OUTPUT_DIR = os.getenv('PIXMIX_OUTPUT_DIR')
YOLO_WITH_AUGMENTATION = bool(os.getenv('YOLO_WITH_AUGMENTATION'))
FASTER_RCNN_WITH_AUGMENTATION = bool(os.getenv('YOLO_WITH_AUGMENTATION'))
TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')


def main():
    try:
        # Check If Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
        does_exist = check_directory_content(input_image_paths + input_annotation_paths)

        # Create Output Directories
        output_image_paths, output_annotation_paths = get_subdirectories(PIXMIX_OUTPUT_DIR)
        create_directories(output_image_paths + output_annotation_paths)

        if does_exist:
            input_image_dir_train = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            input_annotation_dir_train = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)
            output_image_dir_train = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, TRAIN_PATH)
            output_annotation_dir_train = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, TRAIN_PATH)

            input_image_dir_val = os.path.join(PREPROCESSING_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            input_annotation_dir_val = os.path.join(PREPROCESSING_OUTPUT_DIR, LABEL_PATH, VAL_PATH)
            output_image_dir_val = os.path.join(PIXMIX_OUTPUT_DIR, IMAGES_PATH, VAL_PATH)
            output_annotation_dir_val = os.path.join(PIXMIX_OUTPUT_DIR, LABEL_PATH, VAL_PATH)

            if YOLO_WITH_AUGMENTATION or FASTER_RCNN_WITH_AUGMENTATION:
                # Handle Training Data
                pixmix(input_image_dir_train, output_image_dir_train, input_annotation_dir_train, output_annotation_dir_train)

                for img in os.listdir(input_image_dir_val):
                    shutil.copy2(os.path.join(input_image_dir_val, img), os.path.join(output_image_dir_val, img))

                for ann in os.listdir(input_annotation_dir_val):
                    shutil.copy2(os.path.join(input_annotation_dir_val, ann),
                                 os.path.join(output_annotation_dir_val, ann))

    except ValueError as e:
        print(f"Error occurred during augmentation: {e}")


if __name__ == '__main__':
    main()
