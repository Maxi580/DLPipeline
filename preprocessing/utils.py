import os

import yaml

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')
DATA_SPLIT_DIRECTORY = [TRAIN_PATH, VAL_PATH]


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def get_subdirectories(directory, second_subdirectory):
    """Appends val and train to and returns these paths"""
    image_subdirectories = []
    annotation_subdirectories = []
    for split_sub_directory in DATA_SPLIT_DIRECTORY:
        image_subdirectories.append(os.path.join(directory, IMAGES_PATH, split_sub_directory))
        annotation_subdirectories.append(os.path.join(directory, second_subdirectory, split_sub_directory))
    return image_subdirectories, annotation_subdirectories


def check_directory_content(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
            return False
        elif not os.listdir(path):
            print(f"Warning: Directory is empty: {path}")
            return False
    return True


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_or_load_class_mapping(mapping_file):
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = yaml.safe_load(f)
            return {v: int(k) for k, v in mapping.items()}
    return {}


def save_class_mapping(mapping_file, class_mapping):
    yaml_mapping = {str(class_id): class_name for class_name, class_id in class_mapping.items()}
    with open(mapping_file, 'w') as f:
        yaml.safe_dump(yaml_mapping, f, default_flow_style=False)


def parse_txt_annotation(annotation):
    parts = annotation.split()
    if len(parts) != 5:
        raise ValueError("Invalid annotation format")
    return int(parts[0])  # Return the class ID


def create_class_mapping(class_ids):
    return {f"class{id + 1}": id for id in class_ids}


def process_txt_files(directory):
    class_ids = set()
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                for line in f:
                    try:
                        class_id = parse_txt_annotation(line.strip())
                        class_ids.add(class_id)
                    except ValueError:
                        print(f"Skipping invalid line in {filename}: {line.strip()}")
    return sorted(class_ids)
