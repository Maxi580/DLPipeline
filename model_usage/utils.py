import os


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


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
