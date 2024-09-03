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



def is_directory_empty(directory_path):
    try:
        return len(os.listdir(directory_path)) == 0
    except Exception as e:
        print(f"Could not check contents of directory {directory_path}: {e}")
        return True


def check_directory_content(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
            return False
        elif is_directory_empty(path):
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
