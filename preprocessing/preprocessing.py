import json
import os.path
import xml.etree.ElementTree as ET
from PIL import Image
from utils import *
import csv

INPUT_DATA_DIR = os.getenv('INPUT_DATA_DIR')
DETECTION_PREPROCESSING_OUTPUT_DIR = os.getenv('DETECTION_PREPROCESSING_OUTPUT_DIR')
SEGMENTATION_PREPROCESSING_OUTPUT_DIR = os.getenv('SEGMENTATION_PREPROCESSING_OUTPUT_DIR')
MAPPING_FILE = os.getenv('MAPPING_FILE')

MASK_PATH = os.getenv('MASK_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')

IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT'))

TXT_YOLO = 'txt-yolo'
XML_PASCALVOC = 'xml-pascalvoc'
JSON_COCO = 'json-coco'
CSV = 'csv'
COORD_VARIATIONS = {
    'x_min': ['x_min', 'xmin', 'x1', 'left', 'XMIN', 'Xmin'],
    'y_min': ['y_min', 'ymin', 'y1', 'top', 'YMIN', 'Ymin'],
    'x_max': ['x_max', 'xmax', 'x2', 'right', 'YMIN', 'Ymin'],
    'y_max': ['y_max', 'ymax', 'y2', 'bottom', 'YMAX', 'Ymax'],
    'width': ['width', 'w'],
    'height': ['height', 'h']
}
CLASS_NAME_VARIATIONS = ['class', 'class_name', 'category', 'label', 'name']


def detect_annotation_format(input_directory, file):
    """Checks for File Type by Extension and filters for annotation format. Annotation Format may not be 100% be
     accurate but is a good indication"""
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
        # Check for Yolo which is common for txt files
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) == 5 and all(is_float(val) for val in first_line):
                if all(0 <= float(val) <= 1 for val in first_line[1:]):
                    return TXT_YOLO
    elif file_extension == '.json':
        # Check for Coco which is common for json files
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'images' in data and 'annotations' in data and 'categories' in data:
                return JSON_COCO
        except json.JSONDecodeError:
            pass

    elif file_extension == '.csv':
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip().split(',')
                second_line = f.readline().strip().split(',')
                # Check 1: Verify the header
                expected_headers = ['filename', 'width', 'height', 'class']

                def find_coordinate_index(headers, variations):
                    for var in variations:
                        if var in headers:
                            return headers.index(var)
                    return -1

                if all(header in first_line for header in expected_headers):
                    # Check 2: Ensure there's at least one data row
                    if len(second_line) == len(first_line):
                        # Check 3: Verify data types and find indices
                        try:
                            width_idx = find_coordinate_index(first_line, COORD_VARIATIONS['width'])
                            height_idx = find_coordinate_index(first_line, COORD_VARIATIONS['height'])
                            class_idx = first_line.index('class')
                            x_min_idx = find_coordinate_index(first_line, COORD_VARIATIONS['x_min'])
                            y_min_idx = find_coordinate_index(first_line, COORD_VARIATIONS['y_min'])
                            x_max_idx = find_coordinate_index(first_line, COORD_VARIATIONS['x_max'])
                            y_max_idx = find_coordinate_index(first_line, COORD_VARIATIONS['y_max'])

                            if all(idx != -1 for idx in
                                   [width_idx, height_idx, x_min_idx, y_min_idx, x_max_idx, y_max_idx]):

                                int(second_line[width_idx])  # width
                                int(second_line[height_idx])  # height
                                float(second_line[x_min_idx])  # x_min
                                float(second_line[y_min_idx])  # y_min
                                float(second_line[x_max_idx])  # x_max
                                float(second_line[y_max_idx])  # y_max

                                # Check 4: Ensure bounding box coordinates are within image dimensions
                                width, height = int(second_line[width_idx]), int(second_line[height_idx])
                                x_min, y_min = float(second_line[x_min_idx]), float(second_line[y_min_idx])
                                x_max, y_max = float(second_line[x_max_idx]), float(second_line[y_max_idx])

                                if 0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height:
                                    # Check 5: Verify class name is non-empty
                                    if second_line[class_idx].strip():
                                        return CSV

                        except ValueError:
                            pass  # Data type check failed
        except (IOError, IndexError):

            pass


def preprocess_txt_yolo_annotation(input_directory, input_file, output_path):
    try:
        file_path = os.path.join(input_directory, input_file)
        with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                outfile.write(line)
    except Exception as e:
        print(f"An unexpected error occurred while copying txt annotations: {e}")


def preprocess_xml_pascalvoc_annotation(xml_folder, xml_file, output_file, class_mapping):
    """ XML Files have PascalVOC Format, which we will transform to =>
        YOLO: <class_id> <x_center> <y_center> <width> <height>"""
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    with open(output_file, 'w') as yolo_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_mapping = create_or_load_class_mapping(MAPPING_FILE)

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)
            class_id = int(class_mapping[class_name])

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
    return class_mapping


def get_json_category_name(category):
    for name_key in CLASS_NAME_VARIATIONS:
        if name_key in category:
            return category[name_key]
    # If no matching key is found, use the first available key or return None
    return next(iter(category.values())) if category else None


def preprocess_json_coco_annotation(input_directory, coco_file, output_path, class_mapping):
    if class_mapping is None:
        class_mapping = {}

    file_path = os.path.join(input_directory, coco_file)
    with open(file_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    for category in coco_data['categories']:
        category_name = category['name']
        if category_name not in class_mapping:
            class_mapping[category_name] = len(class_mapping)

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

        with open(output_path, 'a') as yolo_file:
            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    return class_mapping


def find_coordinate(row, coord_names):
    """Helper Function for csv parsing, to ensure every name in COORD_VARIATIONS is checked"""
    for name in coord_names:
        if name in row:
            return float(row[name])
    raise KeyError(f"Could not find any of {coord_names} in the CSV row")


def get_csv_class_id(row, fieldnames, class_mapping):
    class_name = None
    for class_column in CLASS_NAME_VARIATIONS:
        if class_column in fieldnames:
            class_name = row.get(class_column)
            if class_name:
                break

    if class_name is None:
        raise ValueError(f"No valid class column found. Available columns: {fieldnames}")

    if class_name not in class_mapping:
        class_mapping[class_name] = len(class_mapping)

    return class_mapping[class_name], class_mapping


def preprocess_csv_to_yolo(input_directory, csv_file, output_file, class_mapping):
    if class_mapping is None:
        class_mapping = {}

    file_path = os.path.join(input_directory, csv_file)
    with open(file_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        current_image = None
        yolo_file = None

        for row in csv_reader:
            image_name = row['filename']

            # If we've moved to a new image, close the previous file and open a new one
            if image_name != current_image:
                if yolo_file:
                    yolo_file.close()
                current_image = image_name
                yolo_file = open(output_file, 'w')

            # Get class id
            try:
                class_id, class_mapping = get_csv_class_id(row, csv_reader.fieldnames, class_mapping)
            except ValueError as e:
                print(f"Error processing row {row}: {e}")
                continue

            # Parse bounding box coordinates
            try:
                x_min = find_coordinate(row, COORD_VARIATIONS['x_min'])
                y_min = find_coordinate(row, COORD_VARIATIONS['y_min'])

                # Check if we have explicit x_max and y_max, otherwise calculate from width and height
                try:
                    x_max = find_coordinate(row, COORD_VARIATIONS['x_max'])
                    y_max = find_coordinate(row, COORD_VARIATIONS['y_max'])
                except KeyError:
                    width = find_coordinate(row, COORD_VARIATIONS['width'])
                    height = find_coordinate(row, COORD_VARIATIONS['height'])
                    x_max = x_min + width
                    y_max = y_min + height

            except KeyError as e:
                print(f"Error: Missing coordinate in CSV for {image_name}: {e}")
                continue

            x_center = (x_min + x_max) / (2 * IMAGE_WIDTH)
            y_center = (y_min + y_max) / (2 * IMAGE_HEIGHT)
            width = (x_max - x_min) / IMAGE_WIDTH
            height = (y_max - y_min) / IMAGE_HEIGHT

            x_center = max(0, min(1, int(x_center)))
            y_center = max(0, min(1, int(y_center)))
            width = max(0, min(1, int(width)))
            height = max(0, min(1, int(height)))

            yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if yolo_file:
            yolo_file.close()

    return class_mapping


def format_annotations(input_directory, output_directory):
    """Annotations have different Formats (YOLO/PASCALVOC/COCO) in different filetypes (xml,json,csv)
       This Function attempts to parse it as good as possible and can not have a 100% success rate.
       The core idea is to detect the correct annotation format/file and then parse it.
       It is standard that Yolo is txt, pascalvoc is xml and coco is json, which is why this is checked. For CSV
       its just a general check for the values specified in COORD_VARIATIONS at the top.
       Class mapping is saved which is important for good inference afterwards."""
    for annotation in os.listdir(input_directory):
        name = os.path.splitext(os.path.basename(annotation))[0]
        output_file = os.path.join(output_directory, f"{name}.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        annotation_format = detect_annotation_format(input_directory, annotation)
        class_mapping = create_or_load_class_mapping(MAPPING_FILE)
        if annotation_format == TXT_YOLO:
            preprocess_txt_yolo_annotation(input_directory, annotation, output_file)
            # Class Mapping makes no Sense for YOlo as it has only numbers
        elif annotation_format == XML_PASCALVOC:
            class_mapping = preprocess_xml_pascalvoc_annotation(input_directory, annotation, output_file, class_mapping)
            save_class_mapping(MAPPING_FILE, class_mapping)
        elif annotation_format == JSON_COCO:
            class_mapping = preprocess_json_coco_annotation(input_directory, annotation, output_file, class_mapping)
            save_class_mapping(MAPPING_FILE, class_mapping)
        elif annotation_format == CSV:
            class_mapping = preprocess_csv_to_yolo(input_directory, annotation, output_file, class_mapping)
            save_class_mapping(MAPPING_FILE, class_mapping)
        else:
            raise ValueError(f"Unexpected annotation format: {annotation}")


def resize_annotation(base_name, original_width, original_height, output_dir):
    """Annotations are expected in YOLO txt Format and adjusted to resizing of the corresponding image
       For that we find the correct annotation by assuming that it has the same name as the image
       then we read the annotation (There can be more than just 1 bounding box per image) and adjust coordinates
       The check that values have to be between 0 and 1 is because anything else is illegal in YOLO"""
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


def resize_image(input_path, output_path):
    """Resizes one image, also the images get formated to 3-Channel RGB for consistency. Augmentation will not
       work unless images have 3 Channels. Function is Designed and should work
       for Tiff/tif, jpg/jpeg and png format."""
    with Image.open(input_path) as img:
        original_width, original_height = img.size

        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')

        img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)  # Type: ignore

        output_format = os.path.splitext(output_path)[1].lower()
        if output_format in ['.jpg', '.jpeg']:
            img_resized.save(output_path, 'JPEG')
        elif output_format == '.tiff':
            img_resized.save(output_path, 'TIFF')
        else:
            img_resized.save(output_path)

    return original_width, original_height


def detection_preprocess(input_image_directories, input_annotations_directories):
    """Create Output Directory, Go through every Annotation in Input Directory to change format
        Go through every Picture in input Directory to change Size and find corresponding annotation to adjust it to
        resize"""
    # Create necessary Directories to store Data
    output_subdirectories = get_subdirectories(DETECTION_PREPROCESSING_OUTPUT_DIR, LABEL_PATH)
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


def find_mask(basename, input_directory):
    """Checks if a fitting png mask does exist and returns it"""
    extensions = ['.png', '.jpg', '.jpeg', '.tiff']

    for extension in extensions:
        filepath = os.path.join(input_directory, basename + extension)
        if os.path.isfile(filepath):
            return filepath, extension

    print(f"Warning: Mask for {basename} not found")



def segmentation_preprocess(input_image_directories, input_mask_directories):
    # Create Output Directories
    output_image_directories, output_mask_directories = get_subdirectories(SEGMENTATION_PREPROCESSING_OUTPUT_DIR, MASK_PATH)
    create_directories(output_image_directories + output_mask_directories)

    # Go through every image and if corresponding mask is found, resize both
    for idx, input_image_directory in enumerate(input_image_directories):
        input_image_directory = input_image_directory
        input_mask_directory = input_mask_directories[idx]

        for image_name in os.listdir(input_image_directory):
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            image_input_path = os.path.join(input_image_directories[idx], image_name)
            mask_input_path, extension = find_mask(base_name, input_mask_directory)

            if mask_input_path:
                image_output_path = os.path.join(output_image_directories[idx], image_name)
                mask_output_path = os.path.join(output_mask_directories[idx], base_name + extension)

                resize_image(image_input_path, image_output_path)
                resize_image(mask_input_path, mask_output_path)


def main():
    """Resizes every Image to the specified Width and Height
       Annotations get transformed into standard yolo txt format for consistency
       Annotations need to be adjusted on resizing"""
    try:
        # Check If Detection Training Data is available
        input_image_directories, input_annotation_directories = get_subdirectories(INPUT_DATA_DIR, LABEL_PATH)
        detection_does_exist = check_directory_content(input_image_directories + input_annotation_directories)

        # Check If Segmentation Training Data is available
        input_image_directories, input_mask_directories = get_subdirectories(INPUT_DATA_DIR, MASK_PATH)
        segmentation_does_exist = check_directory_content(input_image_directories + input_mask_directories)

        if detection_does_exist:
            print("Detection Data gets preprocessed")
            detection_preprocess(input_image_directories, input_annotation_directories)
        if segmentation_does_exist:
            print("Segmentation Data gets preprocessed")
            segmentation_preprocess(input_image_directories, input_mask_directories)

    except ValueError as e:
        print(f"Directory Content Check could not be performed: {e}")


if __name__ == '__main__':
    main()
