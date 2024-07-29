import random
from utils import *
import albumentations as A

FRACTAL_PATH = os.getenv('FRACTAL_PATH')
WIDTH = int(os.getenv('IMAGE_WIDTH'))
HEIGHT = int(os.getenv('IMAGE_HEIGHT'))
NUMBER_OF_AUGMENTATION_RUNS = int(os.getenv('NUMBER_OF_AUGMENTATION_RUNS'))
AUGMENTATIONS = os.getenv('AUGMENTATIONS')


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
    class_labels = [ann[0] for ann in annotations]

    aug = random.choice(AUGMENTATIONS)
    aug_with_bbox = A.Compose([aug], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    augmented = aug_with_bbox(image=image_np, bboxes=bboxes, class_labels=class_labels)
    image_aug = augmented['image']
    bboxes_aug = augmented['bboxes']
    labels_aug = augmented['class_labels']

    aug_annotations = [[label] + list(bbox) for label, bbox in zip(labels_aug, bboxes_aug)]

    return Image.fromarray(image_aug), aug_annotations


def apply_pixmix(image, annotation, mixing_set, p_aug=0.6, p_mix=0.3):
    # Apply random augmentation
    if random.random() < p_aug:
        image, annotation = random_augmentation(image, annotation)

    # Mix with a random image from the mixing set
    if random.random() < p_mix:
        mixing_pic_path = random.choice(mixing_set)
        mixing_pic = Image.open(mixing_pic_path)
        alpha = random.uniform(0.2, 0.4)
        image = mix_images(image, mixing_pic, alpha)

    return image, annotation


def pixmix(image_input_dir, image_output_dir, annotation_input_dir, annotation_output_dir):
    # Create the output directories if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)

    mixing_set = load_image_set(FRACTAL_PATH)

    cntr = 0
    while cntr < NUMBER_OF_AUGMENTATION_RUNS:
        cntr += 1
        # Process each image in the input directory + the corresponding annotation
        for image_filename in os.listdir(image_input_dir):
            # Find Image
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_image_path = os.path.join(image_input_dir, image_filename)

                # Find matching Annotation
                image_basename = os.path.splitext(os.path.basename(image_filename))[0]
                annotation_found = False
                for annotation_filename in os.listdir(annotation_input_dir):
                    annotation_basename = os.path.splitext(os.path.basename(annotation_filename))[0]
                    if annotation_basename == image_basename:
                        # Open Image
                        annotation_found = True
                        with Image.open(input_image_path) as image:
                            # Extract annotations from txt file
                            annotations = read_yolo_annotation(os.path.join(annotation_input_dir, annotation_filename))
                            augmented_img, augmented_annotations = apply_pixmix(image, annotations, mixing_set)

                            output_image_filename = f"{cntr}{image_filename}"
                            image_output_image_path = os.path.join(image_output_dir, output_image_filename)
                            augmented_img.save(image_output_image_path)

                            output_annotation_filename = f"{cntr}{annotation_filename}"
                            annotation_output_path = os.path.join(annotation_output_dir, output_annotation_filename)
                            with open(annotation_output_path, 'w') as f:
                                for ann in augmented_annotations:
                                    f.write(' '.join(map(str, ann)) + '\n')

                            break
                if not annotation_found:
                    print(f"Warning Annotation for {image_filename} not found")
