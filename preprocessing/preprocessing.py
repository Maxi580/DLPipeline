import csv
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

CREATE_YOLO_MODEL = os.getenv('CREATE_YOLO_MODEL')

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
DATA_PATH = os.getenv('DATA_PATH')

DATA_IMAGES_DIR = DATA_PATH + IMAGES_PATH
DATA_ANNOTATION_DIR = DATA_PATH + ANNOTATION_PATH

YOLO_OUTPUT_DIR = os.getenv('YOLO_OUTPUT_DIR')
YOLO_OUTPUT_ANNOTATION_DIR = YOLO_OUTPUT_DIR + ANNOTATION_PATH
YOLO_OUTPUT_IMAGE_DIR = YOLO_OUTPUT_DIR + IMAGES_PATH

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

CLASSES = os.getenv('CLASSES')


def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0


def check_directories(paths):
    for path in paths:
        if is_directory_empty(path):
            raise ValueError(f"Error: Directory is empty: {path}")


def is_csv_file(file):
    return file.lower().endswith('.csv')


def is_xml_file(file):
    return file.lower().endswith('.xml')


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def preprocess_yolo_annotation_csv(csv_file, output_file):
    """
    Yolo txt annotation File Format: <class_id> <x> <y> <width> <height>
    """
    with open(csv_file, 'r') as csv_input, open(output_file, 'w') as yolo_file:
        csv_reader = csv.DictReader(csv_input)
        for row in csv_reader:
            class_name = row['class_name']
            if class_name not in CLASSES:
                print(f"Warning: Class '{class_name}' not found in classes list. Skipping this annotation.")
                continue
            class_id = CLASSES.index(class_name)

            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])

            x_center = (x_min + x_max) / 2.0 / IMAGE_WIDTH
            y_center = (y_min + y_max) / 2.0 / IMAGE_HEIGHT
            width = (x_max - x_min) / IMAGE_WIDTH
            height = (y_max - y_min) / IMAGE_HEIGHT

            yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def preprocess_yolo_annotation_xml(folder, xml_file, output_file):
    tree = ET.parse(os.path.join(folder, xml_file))
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


def preprocess_yolo_images(input_dir, output_dir):
    """
    Yolo can handle JPG/JPEG, PNG, BMP, TIFF
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  # type: ignore
                img_resized.save(output_path)


def preprocess_yolo():
    """Resize Images / Extract relevant YOLO Information from annotations and format it correctly"""
    annotation_input_paths = [DATA_ANNOTATION_DIR + TRAIN_PATH, DATA_ANNOTATION_DIR + VAL_PATH]
    annotation_output_paths = [YOLO_OUTPUT_ANNOTATION_DIR + TRAIN_PATH, YOLO_OUTPUT_ANNOTATION_DIR + VAL_PATH]

    image_input_paths = [DATA_IMAGES_DIR + TRAIN_PATH, DATA_IMAGES_DIR + VAL_PATH]
    image_output_paths = [YOLO_OUTPUT_IMAGE_DIR + TRAIN_PATH, YOLO_OUTPUT_IMAGE_DIR + VAL_PATH]

    for i in range(2):
        # Process Annotations
        for file in os.listdir(Path(annotation_input_paths[i])):
            file_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(f"{annotation_output_paths[i]}/{file_name}_yolo.txt")

            if is_csv_file(file):
                preprocess_yolo_annotation_csv(file, output_file)

            if is_xml_file(file):
                preprocess_yolo_annotation_xml(annotation_input_paths[i], file, output_file)

        # Process Images
        preprocess_yolo_images(image_input_paths[i], image_output_paths[i])


def main():
    """Preprocess every Image and Annotation, to fit the according model"""

    # Ensure That every Directory exists and the data directories contain files
    try:
        all_paths = ([DATA_ANNOTATION_DIR + TRAIN_PATH, DATA_ANNOTATION_DIR + VAL_PATH] +
                     [DATA_IMAGES_DIR + TRAIN_PATH, DATA_IMAGES_DIR + VAL_PATH])
        create_directories(all_paths)

        yolo_paths = ([YOLO_OUTPUT_IMAGE_DIR + TRAIN_PATH, YOLO_OUTPUT_IMAGE_DIR + VAL_PATH,
                       YOLO_OUTPUT_ANNOTATION_DIR + TRAIN_PATH, YOLO_OUTPUT_ANNOTATION_DIR + VAL_PATH])
        create_directories(yolo_paths)

        # Check If Training Data is available
        check_directories(all_paths)


    except ValueError as e:
        print(e)

    if CREATE_YOLO_MODEL:
        preprocess_yolo()


main()
