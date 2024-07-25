import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
import augmentation.pixmix_utils as utils

FRACTAL_PATH = os.getenv('FRACTAL_PATH')
IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT'))


def load_image_set(directory_path):
    image_set = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
    ])

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, filename)
            try:
                with Image.open(image_path) as img:
                    img = transform(img)
                    image_set.append(img)
            except IOError:
                print(f"Error loading image: {image_path}")

    return image_set


def pixmix(orig, augmentation_chance=0.5, mixing_chance=0.1, beta=3):
    mixings = utils.mixings
    tensorize = transforms.ToTensor()

    #  50% Chance of normal augmentation
    if np.random.random() < augmentation_chance:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    # Chance of Mixing with fractal
    if np.random.random() < mixing_chance:
        aug_image_copy = tensorize(augment_input(orig))
        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)

    return mixed


def augment_input(image, aug_severity=3):
    aug_list = utils.augmentations_all
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


def apply_pixmix_to_image(image, mixing_set):
    mixing_pic = np.random.choice(mixing_set)
    return pixmix(image, mixing_pic)


if __name__ == '__main__':
    mixing_set = load_image_set(FRACTAL_PATH)
    picture_path = 'C:/Users/a880902/OneDrive - Eviden/Desktop/Datasetss/LicensePlates/images/train'
    image_data_set = load_image_set(picture_path)

    apply_pixmix_to_image(picture_path, mixing_set)
