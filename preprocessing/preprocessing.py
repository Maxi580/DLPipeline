from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from utils import *

CREATE_YOLO_MODEL = os.getenv('CREATE_YOLO_MODEL')

INPUT_DATA_DIR = os.getenv('INPUT_DATA_DIR')
PREPROCESSING_YOLO_OUTPUT_DIR = os.getenv('PREPROCESSING_YOLO_OUTPUT_DIR')


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

CLASSES = os.getenv('CLASSES')


def is_txt_file(file):
    return file.lower().endswith('.txt')


def is_xml_file(file):
    return file.lower().endswith('.xml')


def preprocess_yolo_annotation_txt(input_file, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                outfile.write(line)
    except Exception as e:
        print(f"An unexpected error occurred while copying txt annotations: {e}")


def preprocess_yolo_annotation_xml(xml_folder, xml_file, output_file):
    """YOLO format: <class_id> <x_center> <y_center> <width> <height>"""
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    with open(output_file, 'w') as yolo_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = CLASSES.index(class_name)

            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            x_center = (x_min + x_max) / 2.0 / IMAGE_WIDTH
            y_center = (y_min + y_max) / 2.0 / IMAGE_HEIGHT
            width = (x_max - x_min) / IMAGE_WIDTH
            height = (y_max - y_min) / IMAGE_HEIGHT

            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def preprocess_yolo_images(input_dir, output_dir, annotation_dir):
    """
    Yolo can handle JPG/JPEG, PNG, BMP, TIFF
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            basename = os.path.splitext(filename)[0]
            # Ensuring annotation and Image have same base name

            with Image.open(input_path) as img:
                original_width, original_height = img.size  # Needed to adjust annotation
                img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  # type: ignore
                img_resized.save(output_path)

            # Adjust annotations to resize
            label_path = os.path.join(annotation_dir, f"{basename}.txt")
            if not os.path.exists(label_path):
                print(f"Warning: Annotation file not found: {label_path}. Skipping this image.")
                return

            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                class_id, x, y, w, h = map(float, line.strip().split())

                # Adjust coordinates
                new_x = x * (original_width / IMAGE_WIDTH)
                new_y = y * (original_height / IMAGE_HEIGHT)
                new_w = w * (original_width / IMAGE_WIDTH)
                new_h = h * (original_height / IMAGE_HEIGHT)

                # Ensure values are within [0, 1] range
                new_x = max(0, min(1, new_x))
                new_y = max(0, min(1, new_y))
                new_w = max(0, min(1, new_w))
                new_h = max(0, min(1, new_h))

                new_lines.append(f"{int(class_id)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

            with open(label_path, 'w') as f:
                f.writelines(new_lines)


def preprocess_yolo():
    """Resize Images / Extract relevant YOLO Information from annotations and format it correctly"""
    input_subdirectories = get_subdirectories(INPUT_DATA_DIR)
    input_image_directories = input_subdirectories[0]
    input_annotations_directories = input_subdirectories[1]

    output_subdirectories = get_subdirectories(PREPROCESSING_YOLO_OUTPUT_DIR)
    output_image_directories = output_subdirectories[0]
    output_annotations_directories = output_subdirectories[1]

    for i in range(len(input_annotations_directories)):
        # Process Annotations
        for file in os.listdir(Path(input_annotations_directories[i])):
            annotation_file_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(f"{output_annotations_directories[i]}/{annotation_file_name}.txt")

            if is_txt_file(file):
                preprocess_yolo_annotation_txt(file, output_file)

            if is_xml_file(file):
                preprocess_yolo_annotation_xml(input_annotations_directories[i], file, output_file)

    for i in range(len(input_image_directories)):
        # Process Images (Annotations need to be processed as they are adjusted by image resize)
        preprocess_yolo_images(input_image_directories[i], output_image_directories[i],
                               output_annotations_directories[i])


def main():
    """Preprocess every Image and Annotation, to fit the according model"""
    try:
        # Check If Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(INPUT_DATA_DIR)
        check_directory_content(input_image_paths + input_annotation_paths)
    except ValueError as e:
        print(f"Directory Content Check could not be performed: {e}")

    if CREATE_YOLO_MODEL:
        # Create Directories to store processed YOLO Data
        yolo_image_paths, yolo_annotation_paths = get_subdirectories(PREPROCESSING_YOLO_OUTPUT_DIR)
        create_directories(yolo_image_paths + yolo_annotation_paths)

        preprocess_yolo()


if __name__ == '__main__':
    main()
