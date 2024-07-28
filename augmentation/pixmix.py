import random

from utils import *
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

FRACTAL_PATH = os.getenv('FRACTAL_PATH')
WIDTH = int(os.getenv('IMAGE_WIDTH'))
HEIGHT = int(os.getenv('IMAGE_HEIGHT'))

AUGMENTATIONS = [
    # Geometrical
    iaa.Fliplr(1.0),
    iaa.Flipud(1.0),
    iaa.Affine(rotate=(-15, 15)),
    iaa.Affine(shear=(-20, 20)),

    # Graphical
    iaa.AddToSaturation((-10, 10)),
    iaa.LinearContrast((0.6, 1.4)),
    iaa.AddToHue((-10, 10)),
    iaa.AddToBrightness((-30, 30)),
    iaa.GaussianBlur(sigma=(0, 5.0)),
]


def mix_images(image, fractal, alpha=0.5):
    # Ensure both images have the same size and mode
    if image.size != fractal.size:
        fractal = fractal.resize(image.size, Image.LANCZOS)  # Type: ignore
    if image.mode != fractal.mode:
        fractal = fractal.convert(image.mode)

    return Image.blend(image, fractal, alpha)


def random_augmentation(image, annotations):
    image_np = np.array(image)

    # Convert YOLO annotations to imgaug format
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=(ann[1] - ann[3]/2) * image.width,
                    y1=(ann[2] - ann[4]/2) * image.height,
                    x2=(ann[1] + ann[3]/2) * image.width,
                    y2=(ann[2] + ann[4]/2) * image.height,
                    label=ann[0])
        for ann in annotations
    ], shape=image_np.shape)

    aug = random.choice(AUGMENTATIONS)
    image_aug, bbs_aug = aug(image=image_np, bounding_boxes=bbs)

    # Get dimensions of augmented image
    aug_height, aug_width = image_aug.shape[:2]

    # Convert back to YOLO format
    aug_annotations = []
    for bb in bbs_aug.bounding_boxes:
        x_center = (bb.x1 + bb.x2) / (2 * aug_width)
        y_center = (bb.y1 + bb.y2) / (2 * aug_height)
        width = (bb.x2 - bb.x1) / aug_width
        height = (bb.y2 - bb.y1) / aug_height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
            aug_annotations.append([bb.label, x_center, y_center, width, height])
        else:
            print(f"Warning: Invalid bounding box detected and removed: {[bb.label, x_center, y_center, width, height]}")

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

                        output_filename = f"{image_filename}"

                        image_output_image_path = os.path.join(image_output_dir, output_filename)
                        augmented_img.save(image_output_image_path)

                        annotation_output_path = os.path.join(annotation_output_dir, annotation_filename)
                        with open(annotation_output_path, 'w') as f:
                            for ann in augmented_annotations:
                                f.write(' '.join(map(str, ann)) + '\n')

                        break
            if not annotation_found:
                print(f"Warning Annotation for {image_filename} not found")
