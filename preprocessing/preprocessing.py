from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from utils import *

INPUT_DATA_DIR = os.getenv('INPUT_DATA_DIR')
PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')

IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT'))
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


def preprocess_xml_annotation(xml_folder, xml_file, output_file):
    """Used format: YOLO (because its standard for txt): <class_id> <x_center> <y_center> <width> <height>"""
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    name = os.path.splitext(os.path.basename(xml_file))[0]
    output_file = os.path.join(output_file, name + '.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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


def format_annotations(input_directory, output_directory):
    for annotation in os.listdir(input_directory):
        if is_txt_file(annotation):
            pass
        elif is_xml_file(annotation):
            preprocess_xml_annotation(input_directory, annotation, output_directory)


def resize_annotation(base_name, original_width, original_height, output_dir):
    """Adjust annotations to image resize"""
    annotation_path = os.path.join(output_dir, f"{base_name}.txt")

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found: {annotation_path}. Skipping this image.")
        return

    with open(annotation_path, 'r') as f:
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

    with open(annotation_path, 'w') as f:
        f.writelines(new_lines)


def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        original_width, original_height = img.size
        img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  #type: ignore
        img_resized.save(output_path)
    return original_width, original_height


def preprocess():
    """Resizes Images to 640/640 (defined in .env) and adjusts bounding boxes accordingly"""
    input_subdirectories = get_subdirectories(INPUT_DATA_DIR)
    input_image_directories = input_subdirectories[0]
    input_annotations_directories = input_subdirectories[1]

    # Create necessary Directories to store Data
    output_subdirectories = get_subdirectories(PREPROCESSING_OUTPUT_DIR)
    output_image_directories = output_subdirectories[0]
    output_annotations_directories = output_subdirectories[1]
    create_directories(output_image_directories + output_annotations_directories)

    # Format Annotations to standard YOLO txt files and move them
    for idx, input_annotation_directory in enumerate(input_annotations_directories):
        format_annotations(input_annotations_directories[idx], output_annotations_directories[idx])

    # Resize every Picture and adjust annotation
    for idx, input_image_directory in enumerate(input_image_directories):
        for image_name in os.listdir(input_image_directories[idx]):
            base_name = os.path.splitext(os.path.basename(image_name))[0]

            image_input_path = os.path.join(input_image_directories[idx], image_name)
            image_output_path = os.path.join(output_image_directories[idx], image_name)
            original_width, original_height = resize_image(image_input_path, image_output_path)

            resize_annotation(base_name, original_width, original_height, output_annotations_directories[idx])


def main():
    """Resize every Image and adjust Annotation"""
    try:
        # Check If Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(INPUT_DATA_DIR)
        check_directory_content(input_image_paths + input_annotation_paths)
    except ValueError as e:
        print(f"Directory Content Check could not be performed: {e}")

    preprocess()


if __name__ == '__main__':
    main()
