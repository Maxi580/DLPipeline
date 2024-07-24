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

CSV_CLASS_NAMES = os.getenv('CSV_CLASS_NAMES')
XMIN_NAMES = os.getenv('XMIN_NAMES')
YMIN_NAMES = os.getenv('YMIN_NAMES')
XMAX_NAMES = os.getenv('XMAX_NAMES')
YMAX_NAMES = os.getenv('YMAX_NAMES')

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


def get_column_name(header, possible_names):
    header_lower = [h.lower() for h in header]
    for name in possible_names:
        if name.lower() in header_lower:
            return header[header_lower.index(name.lower())]
    raise ValueError(f"Could not find any of these columns: {possible_names}")


def preprocess_yolo_annotation_csv(csv_file, output_file):
    """
    Yolo txt annotation File Format: <class_id> <x> <y> <width> <height>
    """
    with open(csv_file, 'r') as csv_input, open(output_file, 'w') as yolo_file:
        csv_reader = csv.DictReader(csv_input)
        header = csv_reader.fieldnames

        # Get column names, if defined in possible Column Names
        class_col = get_column_name(header, CSV_CLASS_NAMES)
        xmin_col = get_column_name(header, XMIN_NAMES)
        ymin_col = get_column_name(header, YMIN_NAMES)
        xmax_col = get_column_name(header, XMAX_NAMES)
        ymax_col = get_column_name(header, YMAX_NAMES)

        # Check if all required columns are found
        if not all([class_col, xmin_col, ymin_col, xmax_col, ymax_col]):
            missing_columns = [name for name, col in [
                ("Class", class_col), ("X Min", xmin_col), ("Y Min", ymin_col),
                ("X Max", xmax_col), ("Y Max", ymax_col)
            ] if col is None]
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. Change possible column "
                             f"names in .env file.")

        for row in csv_reader:
            class_name = row[class_col]
            if class_name not in CLASSES:
                print(f"Warning: Class '{class_name}' not found in classes list. Skipping this annotation.")
                continue
            class_id = CLASSES.index(class_name)

            x_min = float(row[xmin_col])
            y_min = float(row[ymin_col])
            x_max = float(row[xmax_col])
            y_max = float(row[ymax_col])

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


def preprocess_yolo_images(input_dir, output_dir, annotation_file_path):
    """
    Yolo can handle JPG/JPEG, PNG, BMP, TIFF
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # Ensuring annotation and Image have same base name

            with Image.open(input_path) as img:
                original_width, original_height = img.size  # Needed to adjust annotation
                img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  # type: ignore
                img_resized.save(output_path)

            # Adjust annotations to resize
            label_path = os.path.join(annotation_file_path, filename)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Annotation file not found: {label_path}. Please make Sure annotations"
                                        f"have the same Name as Images.")

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
    annotation_input_paths = [DATA_ANNOTATION_DIR + TRAIN_PATH, DATA_ANNOTATION_DIR + VAL_PATH]
    annotation_output_paths = [YOLO_OUTPUT_ANNOTATION_DIR + TRAIN_PATH, YOLO_OUTPUT_ANNOTATION_DIR + VAL_PATH]

    image_input_paths = [DATA_IMAGES_DIR + TRAIN_PATH, DATA_IMAGES_DIR + VAL_PATH]
    image_output_paths = [YOLO_OUTPUT_IMAGE_DIR + TRAIN_PATH, YOLO_OUTPUT_IMAGE_DIR + VAL_PATH]

    for i in range(2):
        for file in os.listdir(Path(annotation_input_paths[i])):
            # Process Annotations
            annotation_file_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(f"{annotation_output_paths[i]}/{annotation_file_name}.txt")

            if is_csv_file(file):
                preprocess_yolo_annotation_csv(file, output_file)

            if is_xml_file(file):
                preprocess_yolo_annotation_xml(annotation_input_paths[i], file, output_file)

        # Process Images
        preprocess_yolo_images(image_input_paths[i], image_output_paths[i], annotation_output_paths[i])


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
