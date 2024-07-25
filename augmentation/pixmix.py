import random
import numpy as np
import os
from PIL import Image
from augmentation_utils import *
from mix_utils import *
from utils import *

FRACTAL_PATH = os.getenv('FRACTAL_PATH')
WIDTH = os.getenv('IMAGE_WIDTH')
HEIGHT = os.getenv('IMAGE_HEIGHT')


def apply_random_augmentation(image):
    augmentation_functions = [
        flip_horizontal,
        rotate,
        adjust_brightness,
        adjust_contrast,
        adjust_saturation,
        add_noise,
        blur,
        sharpen
    ]

    aug_func = random.choice(augmentation_functions)
    return aug_func(image)


def apply_pixmix(image, mixing_set, p_aug=0.5, p_mix=0.3):
    # Apply random augmentation
    if random.random() < p_aug:
        image = apply_random_augmentation(image)

    # Mix with a random image from the mixing set
    if random.random() < p_mix:
        mixing_pic_path = random.choice(mixing_set)
        mixing_pic = Image.open(mixing_pic_path)
        alpha = random.uniform(0.2, 0.4)
        image = mix_images(image, mixing_pic, alpha)

    return image


def process_and_save_images(input_path, output_path, mixing_set):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_image_path = os.path.join(input_path, filename)
            with Image.open(input_image_path) as img:
                augmented_img = apply_pixmix(img, mixing_set)

                output_filename = f"augmented_{filename}"
                output_image_path = os.path.join(output_path, output_filename)
                augmented_img.save(output_image_path)


if __name__ == '__main__':
    mixing_set = load_image_set('C:/Users/maxie/Desktop/DLPipeline/augmentation/fractals')
    picture_path = 'C:/Users/maxie/Desktop/LicensePlateData/images/train'
    output_path = 'C:/Users/maxie/Desktop/LicensePlateData/images/augmented'

    process_and_save_images(picture_path, output_path, mixing_set)