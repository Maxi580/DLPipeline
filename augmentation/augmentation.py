import numpy as np
from PIL import Image
from utils import *

from aug_albumentations import augmentation

ALBUMENTATION_YOLO_OUTPUT_DIR = os.getenv('ALBUMENTATION_YOLO_OUTPUT_DIR')

NUM_AUGMENTATIONS = int(os.getenv('NUM_AUGMENTATIONS'))
ALBUMENTATION = os.getenv('ALBUMENTATION')

CREATE_YOLO_MODEL = os.getenv('CREATE_YOLO_MODEL')
PREPROCESSING_YOLO_OUTPUT_DIR = os.getenv('PREPROCESSING_YOLO_OUTPUT_DIR')


def yolo_augmentation():
    # Check if processed YOLO Data is available
    yolo_paths = get_subdirectories(PREPROCESSING_YOLO_OUTPUT_DIR)
    yolo_image_directories = yolo_paths[0]
    yolo_annotation_directories = yolo_paths[1]

    try:
        check_directory_content(yolo_image_directories + yolo_annotation_directories)
    except ValueError as e:
        print(f"Couldn't Find Preprocessed YOLO Data: {e}")

    # Get every Image, search corresponding annotation, augment, save to new Folder
    for i in range(len(yolo_image_directories)):
        for image_name in os.listdir(yolo_image_directories[i]):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(yolo_image_directories[i], image_name)

                basename = os.path.splitext(os.path.basename(image_name))[0]
                annotation_name = f"{basename}.txt"
                annotation_path = os.path.join(yolo_annotation_directories[i] + '/' + annotation_name)

                if not os.path.exists(annotation_path):
                    print(f"Warning: Annotation file not found: {annotation_path}. Skipping this image.")
                    continue

                if ALBUMENTATION:
                    try:
                        augmentation_paths = get_subdirectories(ALBUMENTATION_YOLO_OUTPUT_DIR)
                        create_directories(augmentation_paths[0] + augmentation_paths[1])
                    except ValueError as e:
                        print(f"Couldn't Create Directories: {e}")

                    albumentation_paths = get_subdirectories(ALBUMENTATION_YOLO_OUTPUT_DIR)
                    albumentations_image_directories = albumentation_paths[0]
                    albumentations_annotation_directories = albumentation_paths[1]

                    for number_of_augmentations in range(NUM_AUGMENTATIONS):  # Todo ???
                        bbox, class_labels = read_yolo_file(annotation_path)
                        bbox_format = 'yolo'
                        with Image.open(image_path) as img:
                            image_np = np.array(img)

                        augmented_image, augmented_bbox, augmented_label = augmentation(image_np, bbox, class_labels, bbox_format)

                        save_yolo_file(albumentations_annotation_directories[i], annotation_name,
                                       augmented_bbox, augmented_label)

                        augmented_pil_image = Image.fromarray(augmented_image.astype('uint8'))
                        output_image_path = os.path.join(albumentations_image_directories[i], image_name)
                        augmented_pil_image.save(output_image_path)


def main():
    if CREATE_YOLO_MODEL:
        yolo_augmentation()


if __name__ == '__main__':
    main()
