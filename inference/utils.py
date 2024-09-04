import os


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def check_directory_content(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Directory does not exist: {path}")
            return False
        elif not os.listdir(path):
            print(f"Warning: Directory is empty: {path}")
            return False
    return True
