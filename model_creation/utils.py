import os

TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
IMAGES_PATH = os.getenv('IMAGES_PATH')
LABEL_PATH = os.getenv('LABEL_PATH')
DATA_SPLIT_DIRECTORY = [TRAIN_PATH, VAL_PATH]


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_subdirectories(directory, second_subdirectory):
    """Appends val and train to and returns these paths"""
    image_subdirectories = []
    annotation_subdirectories = []
    for split_sub_directory in DATA_SPLIT_DIRECTORY:
        image_subdirectories.append(os.path.join(directory, second_subdirectory, split_sub_directory))
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

