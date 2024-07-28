import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from utils import *

INPUT_DATA_DIR = os.getenv('INPUT_DATA_DIR')
PREPROCESSING_OUTPUT_DIR = os.getenv('PREPROCESSING_OUTPUT_DIR')

IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT'))
CLASSES = os.getenv('CLASSES')

TXT_YOLO = 'txt-yolo'
XML_PASCALVOC = 'xml-pascalvoc'
JSON_COCO = 'json-coco'


def detect_annotation_format(input_directory, file):
    file_extension = os.path.splitext(file)[1].lower()
    file_path = os.path.join(input_directory, file)

    if file_extension == '.xml':
        # Check for PascalVoc which is common for xml
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            if root.tag == 'annotation':
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        if all(bndbox.find(coord) is not None for coord in ['xmin', 'ymin', 'xmax', 'ymax']):
                            return XML_PASCALVOC
        except ET.ParseError:
            pass
    elif file_extension == '.txt':
        # Check for Yolo which is common for txt
        with open(file, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) == 5 and all(is_float(val) for val in first_line):
                return TXT_YOLO
    elif file_extension == '.json':
        # Check for Coco which is common for json files
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            if 'images' in data and 'annotations' in data and 'categories' in data:
                return JSON_COCO
        except json.JSONDecodeError:
            pass


def preprocess_txt_yolo_annotation(input_file, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                outfile.write(line)
    except Exception as e:
        print(f"An unexpected error occurred while copying txt annotations: {e}")


def preprocess_xml_pascalvoc_annotation(xml_folder, xml_file, output_file):
    """ XML Files have PascalVOC Format, which we will transform to =>
        YOLO: <class_id> <x_center> <y_center> <width> <height>"""
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


def preprocess_json_coco_annotation(coco_file, output_folder):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    for ann in coco_data['annotations']:
        image = images[ann['image_id']]
        image_width = image['width']
        image_height = image['height']

        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        width = bbox[2] / image_width
        height = bbox[3] / image_height

        class_id = ann['category_id'] - 1

        output_file = os.path.join(output_folder, f"{image['file_name'].split('.')[0]}.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'a') as yolo_file:
            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def format_annotations(input_directory, output_directory):
    for annotation in os.listdir(input_directory):
        output_file = os.path.join(output_directory)

        annotation_format = detect_annotation_format(input_directory, annotation)
        if annotation_format == TXT_YOLO:
            preprocess_txt_yolo_annotation(annotation, output_file)
        elif annotation_format == XML_PASCALVOC:
            preprocess_xml_pascalvoc_annotation(input_directory, annotation, output_file)
        elif annotation_format == JSON_COCO:
            preprocess_json_coco_annotation(annotation, output_directory)
        else:
            raise ValueError(f"Unexpected annotation format: {annotation}")


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
        new_x = x * (IMAGE_WIDTH / original_width)
        new_y = y * (IMAGE_HEIGHT / original_height)
        new_w = w * (IMAGE_WIDTH / original_width)
        new_h = h * (IMAGE_HEIGHT / original_height)

        # Ensure values are within [0, 1] range
        new_x = max(0, min(1, new_x))
        new_y = max(0, min(1, new_y))
        new_w = max(0, min(1, new_w))
        new_h = max(0, min(1, new_h))

        new_lines.append(f"{int(class_id)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

    with open(annotation_path, 'w') as f:
        f.writelines(new_lines)


def format_image(input_path, output_path):
    """Resizes images and ensures RGB format, properly handling palette images with transparency"""
    with Image.open(input_path) as img:
        original_width, original_height = img.size

        # Check if the image is a palette image with transparency
        if img.mode == 'P' and 'transparency' in img.info:
            # Convert to RGBA first to properly handle transparency
            img = img.convert('RGBA')

        # Now convert to RGB, which will handle the alpha channel correctly
        img = img.convert('RGB')

        img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  # Type: ignore
        output_path = os.path.splitext(output_path)[0] + '.png'
        img_resized.save(output_path, 'PNG', icc_profile=None)
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
            original_width, original_height = format_image(image_input_path, image_output_path)

            resize_annotation(base_name, original_width, original_height, output_annotations_directories[idx])


def main():
    """Resize every Image and adjust Annotation"""
    try:
        # Check If Training Data is available
        input_image_paths, input_annotation_paths = get_subdirectories(INPUT_DATA_DIR)
        does_exist = check_directory_content(input_image_paths + input_annotation_paths)

        if does_exist:
            preprocess()
    except ValueError as e:
        print(f"Directory Content Check could not be performed: {e}")


if __name__ == '__main__':
    main()
