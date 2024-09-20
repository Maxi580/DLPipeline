import random

import numpy as np

from utils import *
import albumentations as A

FRACTAL_PATH = os.getenv('FRACTAL_PATH')
WIDTH = int(os.getenv('IMAGE_WIDTH'))
HEIGHT = int(os.getenv('IMAGE_HEIGHT'))
NUMBER_OF_AUGMENTED_IMAGES = int(os.getenv('NUMBER_OF_AUGMENTED_IMAGES'))
AUGMENTATIONS = os.getenv('AUGMENTATIONS')
PIXMIX_AUGMENTATION_PROBABILITY = float(os.getenv('AUGMENTATION_PROBABILITY'))
PIXMIX_MIXING_PROBABILITY = float(os.getenv('MIXING_PROBABILITY'))
PIXMIX_MIXING_FACTOR_LOWER_RANGE = float(os.getenv('MIXING_FACTOR_LOWER_RANGE'))
PIXMIX_MIXING_FACTOR_UPPER_RANGE = float(os.getenv('MIXING_FACTOR_UPPER_RANGE'))


def get_env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() == 'true'


def get_env_float(name, default=0.0):
    return float(os.getenv(name, default))


def get_env_int(name, default=0):
    return int(os.getenv(name, default))


geometrical_augmentations = []
graphical_augmentations = []


def get_augmentations():
    return geometrical_augmentations + graphical_augmentations


def get_graphical_augmentations():
    return geometrical_augmentations


def get_geometrical_augmentations():
    return geometrical_augmentations


def update_augmentations(geometrical_augmentations, graphical_augmentations):
    """Read every Augmentation Configuration from .env and update it. The Result is a valid Albumentations list."""
    if get_env_bool('ENABLE_HORIZONTAL_FLIP'):
        geometrical_augmentations.append(A.HorizontalFlip(p=1.0))

    if get_env_bool('ENABLE_VERTICAL_FLIP'):
        geometrical_augmentations.append(A.VerticalFlip(p=1.0))

    if get_env_bool('ENABLE_ROTATE'):
        rotate_limit = get_env_int('ROTATE_LIMIT', 45)
        geometrical_augmentations.append(A.Rotate(limit=rotate_limit, p=1.0))

    if get_env_bool('ENABLE_HUE_SATURATION'):
        hue_shift = get_env_int('HUE_SHIFT_LIMIT', 10)
        sat_shift = get_env_int('SAT_SHIFT_LIMIT', 10)
        val_shift = get_env_int('VAL_SHIFT_LIMIT', 10)
        graphical_augmentations.append(A.HueSaturationValue(hue_shift_limit=hue_shift,
                                                            sat_shift_limit=sat_shift,
                                                            val_shift_limit=val_shift,
                                                            p=1.0))

    if get_env_bool('ENABLE_BRIGHTNESS_CONTRAST'):
        brightness_limit = get_env_float('BRIGHTNESS_LIMIT', 0.2)
        contrast_limit = get_env_float('CONTRAST_LIMIT', 0.2)
        graphical_augmentations.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                                                  contrast_limit=contrast_limit,
                                                                  p=1.0))
    if get_env_bool('ENABLE_SHEAR'):
        shear_degree_limit = get_env_float('SHEAR_DEGREE_LIMIT', 20)
        geometrical_augmentations.append(A.Affine(shear=(-shear_degree_limit, shear_degree_limit), p=1.0))

    if get_env_bool('ENABLE_GAUSSIAN_BLUR'):
        blur_min = get_env_int('GAUSSIAN_BLUR_MINIMUM', 3)
        blur_max = get_env_int('GAUSSIAN_BLUR_MAX', 5)
        graphical_augmentations.append(A.GaussianBlur(blur_limit=(blur_min, blur_max), p=1.0))

    if get_env_bool('ENABLE_GAUSSIAN_NOISE'):
        var_limit = get_env_float('NOISE_VAR_LIMIT', 0.05)
        graphical_augmentations.append(A.GaussNoise(var_limit=var_limit, p=1.0))

    if get_env_bool('ENABLE_RANDOM_GAMMA'):
        gamma_limit = get_env_int('RANDOM_GAMMA_LIMIT', 1)
        graphical_augmentations.append(A.RandomGamma(gamma_limit=gamma_limit, p=1.0))

    if get_env_bool('ENABLE_RANDOM_RAIN'):
        graphical_augmentations.append(A.RandomRain(p=1.0))

    if get_env_bool('ENABLE_RANDOM_FOG'):
        graphical_augmentations.append(A.RandomFog(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SNOW'):
        graphical_augmentations.append(A.RandomSnow(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SHADOW'):
        graphical_augmentations.append(A.RandomShadow(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SUNFLARE'):
        graphical_augmentations.append(A.RandomSunFlare(p=1.0))


def mix_images(image, fractal, alpha):
    # Ensure both images have the same size and mode
    if image.size != fractal.size:
        fractal = fractal.resize(image.size, Image.LANCZOS)  # Type: ignore
    if image.mode != fractal.mode:
        fractal = fractal.convert(image.mode)

    return Image.blend(image, fractal, alpha)


def detection_geometrical_augmentation(image, annotations):
    """Using Albumentations for augmentation, annotations get adjusted automatically"""
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Convert YOLO format to COCO format
    bboxes = []
    class_labels = []
    for ann in annotations:
        class_id, x_center, y_center, bbox_width, bbox_height = ann
        x_min = int((x_center - bbox_width / 2) * width)
        y_min = int((y_center - bbox_height / 2) * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)
        bboxes.append([x_min, y_min, bbox_width, bbox_height])
        class_labels.append(int(class_id))

    aug = random.choice(get_geometrical_augmentations())
    aug_with_bbox = A.Compose([aug], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    try:
        augmented = aug_with_bbox(image=image_np, bboxes=bboxes, class_labels=class_labels)

        image_aug = augmented['image']
        bboxes_aug = augmented['bboxes']
        labels_aug = augmented['class_labels']

        # Convert COCO format back to YOLO format
        aug_height, aug_width = image_aug.shape[:2]
        aug_annotations = []
        for bbox, label in zip(bboxes_aug, labels_aug):
            x_min, y_min, bbox_width, bbox_height = bbox
            x_center = (x_min + bbox_width / 2) / aug_width
            y_center = (y_min + bbox_height / 2) / aug_height
            width = bbox_width / aug_width
            height = bbox_height / aug_height
            aug_annotations.append([int(label), x_center, y_center, width, height])

        return Image.fromarray(image_aug), aug_annotations

    except ValueError as e:
        print(f"Augmentation failed: {e}")
        return image, annotations


def segmentation_geometrical_augmentation(image, mask):
    """Applies geometrical Augmentations to image and Mask used for Segmentation"""
    image_np = np.array(image)
    mask_np = np.array(mask)

    aug = random.choice(get_geometrical_augmentations())
    aug_with_bbox = A.Compose([aug])

    augmented_image = aug_with_bbox(image=image_np, mask=mask_np)
    image_aug = augmented_image['image']
    mask_aug = augmented_image['mask']

    return Image.fromarray(image_aug), Image.fromarray(mask_aug)


def graphical_augmentation(image):
    """Applies graphical Augmentations to an image used for Segmentation"""
    image_np = np.array(image)

    aug = random.choice(get_graphical_augmentations())
    aug_with_bbox = A.Compose([aug])

    augmented_image = aug_with_bbox(image=image_np)
    image_aug = augmented_image['image']

    return Image.fromarray(image_aug)


def apply_fractal(image, mixing_set):
    # Mix with a random image from the mixing set
    if random.random() <= PIXMIX_MIXING_PROBABILITY:
        mixing_pic_path = random.choice(mixing_set)
        mixing_pic = Image.open(mixing_pic_path)
        alpha = random.uniform(PIXMIX_MIXING_FACTOR_LOWER_RANGE, PIXMIX_MIXING_FACTOR_UPPER_RANGE)
        image = mix_images(image, mixing_pic, alpha)
    return image


def apply_detection_pixmix(image, annotation, mixing_set, geometrical_probability):
    """Augmentation inspired by Dreamlike Pixmix Repo"""
    # Apply random augmentation
    if random.random() <= PIXMIX_AUGMENTATION_PROBABILITY:
        if random.random() <= geometrical_probability:
            image, annotation = detection_geometrical_augmentation(image, annotation)
        else:
            image = graphical_augmentation(image)

    image = apply_fractal(image, mixing_set)

    return image, annotation


def apply_segmentation_pixmix(image, mask, mixing_set, geometrical_probability):
    """Augmentation, but geometrical augmentations also get applied to the mask"""
    print("Segmentation Pixmix gets applied")
    # Randomly apply augmentation
    if random.random() <= PIXMIX_MIXING_PROBABILITY:
        if random.random() <= geometrical_probability:
            image, mask = segmentation_geometrical_augmentation(image, mask)
        else:
            image = graphical_augmentation(image)

    image = apply_fractal(image, mixing_set)

    return image, mask


def pixmix_setup(image_output_dir, annotation_output_dir):
    # Create the output directories if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)

    update_augmentations(geometrical_augmentations, graphical_augmentations)

    return load_image_set(FRACTAL_PATH)


def detection_pixmix(image_input_dir, image_output_dir, annotation_input_dir, annotation_output_dir):
    """Main augmentation function for Image Detection. Goes through all images, finds annotations.
       Augments Images and adjusts annotations accordingly  (To geographical Changes in Base image)."""
    mixing_set = pixmix_setup(image_output_dir, annotation_output_dir)
    geometrical_probability = len(get_geometrical_augmentations()) / len(get_augmentations())

    list_of_images = os.listdir(image_input_dir)
    list_of_annotations = os.listdir(annotation_input_dir)
    cntr = 0
    # Process each image in the input directory + the corresponding annotation
    while cntr <= NUMBER_OF_AUGMENTED_IMAGES:
        for image_filename in list_of_images:
            # Find Image
            if cntr <= NUMBER_OF_AUGMENTED_IMAGES:
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_image_path = os.path.join(image_input_dir, image_filename)

                    # Find matching Annotation
                    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
                    annotation_found = False
                    for annotation_filename in list_of_annotations:
                        annotation_basename = os.path.splitext(os.path.basename(annotation_filename))[0]
                        if annotation_basename == image_basename:
                            annotation_found = True
                            with Image.open(input_image_path) as image:
                                # Extract annotations from txt file
                                annotations = read_yolo_annotation(
                                    os.path.join(annotation_input_dir, annotation_filename))
                                augmented_img, augmented_annotations = apply_detection_pixmix(image, annotations,
                                                                                              mixing_set,
                                                                                              geometrical_probability)

                                output_image_filename = f"{cntr}.{image_filename}"
                                image_output_path = os.path.join(image_output_dir, output_image_filename)
                                augmented_img.save(image_output_path)

                                output_annotation_filename = f"{cntr}.{annotation_filename}"
                                annotation_output_path = os.path.join(annotation_output_dir, output_annotation_filename)
                                with open(annotation_output_path, 'w') as f:
                                    for ann in augmented_annotations:
                                        f.write(' '.join(map(str, ann)) + '\n')
                                cntr += 1
                                break
                    if not annotation_found:
                        print(f"Warning Annotation for {image_filename} not found")
            else:
                break


def segmentation_pixmix(image_input_dir, image_output_dir, mask_input_dir, mask_output_dir):
    """Main function for segmentation Augmentation. Goes through all images, and also applies geographical Changes
       to the mask. It's very similar. Repetitive Code might be cleaned up sometime."""
    mixing_set = pixmix_setup(image_output_dir, mask_output_dir)
    geometrical_probability = len(get_geometrical_augmentations()) / len(get_augmentations())

    list_of_images = os.listdir(image_input_dir)
    list_of_masks = os.listdir(mask_input_dir)
    cntr = 0
    # Process each image in the input directory + the corresponding mask
    while cntr <= NUMBER_OF_AUGMENTED_IMAGES:
        for image_filename in list_of_images:
            # Find Image
            if cntr <= NUMBER_OF_AUGMENTED_IMAGES:
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_image_path = os.path.join(image_input_dir, image_filename)

                    # Find matching Mask
                    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
                    mask_found = False
                    for mask_filename in list_of_masks:
                        mask_basename = os.path.splitext(os.path.basename(mask_filename))[0]
                        if mask_basename == image_basename:
                            mask_found = True
                            input_mask_path = os.path.join(mask_input_dir, mask_filename)
                            with Image.open(input_image_path) as image:
                                with Image.open(input_mask_path) as mask:
                                    augmented_image, augmented_mask = apply_segmentation_pixmix(image, mask,
                                                                                                mixing_set,
                                                                                                geometrical_probability)

                                    image_output_path = os.path.join(image_output_dir, f"{cntr}.{image_filename}")
                                    augmented_image.save(image_output_path)

                                    mask_output_path = os.path.join(mask_output_dir, f"{cntr}.{mask_filename}")
                                    augmented_mask.save(mask_output_path)

                                    cntr += 1
                                    break

                    if not mask_found:
                        print(f"Warning Mask for {image_filename} not found")
            else:
                break
