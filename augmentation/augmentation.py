from pixmix import *

PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')
PIXMIX_OUTPUT_DIR = os.getenv('PIXMIX_OUTPUT_DIR')
PIXMIX_AUGMENTATION = bool(os.getenv('PIXMIX_AUGMENTATION'))


def main():
    try:
        # Check If Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
        does_exist = check_directory_content(input_image_paths + input_annotation_paths)

        if does_exist:
            output_image_paths, output_annotation_paths = get_subdirectories(PIXMIX_OUTPUT_DIR)

            for idx, input_image_path in enumerate(input_image_paths):
                image_input_dir = input_image_paths[idx]
                image_output_dir = output_image_paths[idx]
                annotation_input_dir = input_annotation_paths[idx]
                annotation_output_dir = output_annotation_paths[idx]

                if PIXMIX_AUGMENTATION:
                    pixmix(image_input_dir, image_output_dir, annotation_input_dir, annotation_output_dir)

    except ValueError as e:
        print(f"Error occurred during augmentation: {e}")


if __name__ == '__main__':
    main()
