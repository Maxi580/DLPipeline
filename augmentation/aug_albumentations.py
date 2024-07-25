import albumentations as A
import numpy as np
import os
import random
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform

FRACTAL_PATH = os.getenv('FRACTAL_PATH')


def fractal_augmentation(image, blend_factor=0.2):
    """Pick a random fractal, resize it to the size of the image, turn into np arrays => blend """
    fractal_dir = FRACTAL_PATH
    fractal_files = [f for f in os.listdir(fractal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    fractal_file = random.choice(fractal_files)
    fractal_path = os.path.join(fractal_dir, fractal_file)

    fractal = Image.open(fractal_path)
    fractal = fractal.resize(image.size)

    image_array = np.array(image)
    fractal_array = np.array(fractal)

    blended_array = (1 - blend_factor) * image_array + blend_factor * fractal_array

    blended_array = blended_array.astype(np.uint8)

    return Image.fromarray(blended_array)


class FractalAugmentation(ImageOnlyTransform):
    def __init__(self, blend_factor=0.2, always_apply=False, p=0.1):
        super(FractalAugmentation, self).__init__(p, always_apply)
        self.blend_factor = blend_factor

    def apply(self, image, **params):
        return np.array(fractal_augmentation(Image.fromarray(image), self.blend_factor))

    def get_transform_init_args_names(self):
        return ("blend_factor",)


def augmentation(image, bboxes, class_labels, bbox_format):
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        #FractalAugmentation(blend_factor=0.2, always_apply=False, p=0.1), Debug
    ], bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels']))

    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return augmented['image'], augmented['bboxes'], augmented['class_labels']
    except Exception as e:
        print(f"Error during augmentation: {e} BBOXES: {bboxes}, IMAGE{image}, Class_LABEL{class_labels}")
        return image, bboxes, class_labels
