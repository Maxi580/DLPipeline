import random
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


augmentations = []


def get_augmentations():
    return augmentations


def update_augmentations(augmentations):
    if get_env_bool('ENABLE_HORIZONTAL_FLIP'):
        augmentations.append(A.HorizontalFlip(p=1.0))

    if get_env_bool('ENABLE_VERTICAL_FLIP'):
        augmentations.append(A.VerticalFlip(p=1.0))

    if get_env_bool('ENABLE_ROTATE'):
        rotate_limit = get_env_int('ROTATE_LIMIT', 45)
        augmentations.append(A.Rotate(limit=rotate_limit, p=1.0))

    if get_env_bool('ENABLE_HUE_SATURATION'):
        hue_shift = get_env_int('HUE_SHIFT_LIMIT', 10)
        sat_shift = get_env_int('SAT_SHIFT_LIMIT', 10)
        val_shift = get_env_int('VAL_SHIFT_LIMIT', 10)
        augmentations.append(A.HueSaturationValue(hue_shift_limit=hue_shift,
                                                  sat_shift_limit=sat_shift,
                                                  val_shift_limit=val_shift,
                                                  p=1.0))

    if get_env_bool('ENABLE_BRIGHTNESS_CONTRAST'):
        brightness_limit = get_env_float('BRIGHTNESS_LIMIT', 0.2)
        contrast_limit = get_env_float('CONTRAST_LIMIT', 0.2)
        augmentations.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                                        contrast_limit=contrast_limit,
                                                        p=1.0))
    if get_env_bool('ENABLE_SHEAR'):
        shear_degree_limit = get_env_float('SHEAR_DEGREE_LIMIT', 20)
        augmentations.append(A.Affine(shear=(-shear_degree_limit, shear_degree_limit), p=1.0))

    if get_env_bool('ENABLE_GAUSSIAN_BLUR'):
        blur_min = get_env_int('GAUSSIAN_BLUR_MINIMUM', 3)
        blur_max = get_env_int('GAUSSIAN_BLUR_MAX', 5)
        augmentations.append(A.GaussianBlur(blur_limit=(blur_min, blur_max), p=1.0))

    if get_env_bool('ENABLE_GAUSSIAN_NOISE'):
        var_limit = get_env_float('NOISE_VAR_LIMIT', 0.05)
        augmentations.append(A.GaussNoise(var_limit=var_limit, p=1.0))

    if get_env_bool('ENABLE_RANDOM_GAMMA'):
        gamma_limit = get_env_int('RANDOM_GAMMA_LIMIT', 1)
        augmentations.append(A.RandomGamma(gamma_limit=gamma_limit, p=1.0))

    if get_env_bool('ENABLE_RANDOM_RAIN'):
        augmentations.append(A.RandomRain(p=1.0))

    if get_env_bool('ENABLE_RANDOM_FOG'):
        augmentations.append(A.RandomFog(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SNOW'):
        augmentations.append(A.RandomSnow(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SHADOW'):
        augmentations.append(A.RandomShadow(p=1.0))

    if get_env_bool('ENABLE_RANDOM_SUNFLARE'):
        augmentations.append(A.RandomSunFlare(p=1.0))


def mix_images(image, fractal, alpha=0.5):
    # Ensure both images have the same size and mode
    if image.size != fractal.size:
        fractal = fractal.resize(image.size, Image.LANCZOS)  # Type: ignore
    if image.mode != fractal.mode:
        fractal = fractal.convert(image.mode)

    return Image.blend(image, fractal, alpha)


def random_augmentation(image, annotations):
    image_np = np.array(image)

    bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]
    class_labels = [int(ann[0]) for ann in annotations]

    aug = random.choice(get_augmentations())
    aug_with_bbox = A.Compose([aug], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    augmented = aug_with_bbox(image=image_np, bboxes=bboxes, class_labels=class_labels)
    image_aug = augmented['image']
    bboxes_aug = augmented['bboxes']
    labels_aug = augmented['class_labels']

    aug_annotations = [[int(label)] + list(bbox) for label, bbox in zip(labels_aug, bboxes_aug)]
    print(aug_annotations)

    return Image.fromarray(image_aug), aug_annotations


def apply_pixmix(image, annotation, mixing_set):
    # Apply random augmentation
    if random.random() < PIXMIX_AUGMENTATION_PROBABILITY:
        image, annotation = random_augmentation(image, annotation)

    # Mix with a random image from the mixing set
    if random.random() < PIXMIX_MIXING_PROBABILITY:
        mixing_pic_path = random.choice(mixing_set)
        mixing_pic = Image.open(mixing_pic_path)
        alpha = random.uniform(PIXMIX_MIXING_FACTOR_LOWER_RANGE, PIXMIX_MIXING_FACTOR_UPPER_RANGE)
        image = mix_images(image, mixing_pic, alpha)

    return image, annotation


def pixmix(image_input_dir, image_output_dir, annotation_input_dir, annotation_output_dir):
    # Create the output directories if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)

    mixing_set = load_image_set(FRACTAL_PATH)
    update_augmentations(augmentations)

    cntr = 0
    # Process each image in the input directory + the corresponding annotation
    while cntr <= NUMBER_OF_AUGMENTED_IMAGES:
        for image_filename in os.listdir(image_input_dir):
            # Find Image
            if cntr <= NUMBER_OF_AUGMENTED_IMAGES:
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_image_path = os.path.join(image_input_dir, image_filename)

                    # Find matching Annotation
                    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
                    annotation_found = False
                    for annotation_filename in os.listdir(annotation_input_dir):
                        annotation_basename = os.path.splitext(os.path.basename(annotation_filename))[0]
                        if annotation_basename == image_basename:
                            annotation_found = True
                            with Image.open(input_image_path) as image:
                                # Extract annotations from txt file
                                annotations = read_yolo_annotation(
                                    os.path.join(annotation_input_dir, annotation_filename))
                                augmented_img, augmented_annotations = apply_pixmix(image, annotations, mixing_set)

                                output_image_filename = f"{cntr}{image_filename}"
                                image_output_image_path = os.path.join(image_output_dir, output_image_filename)
                                augmented_img.save(image_output_image_path)

                                output_annotation_filename = f"{cntr}{annotation_filename}"
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
