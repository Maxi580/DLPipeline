import os
import cv2
import numpy as np
from PIL import Image


def draw_yolo_annotations(image_path, annotation_path, output_path):
    try:
        # Read the image using Pillow
        with Image.open(image_path) as img:
            # Convert to RGB if it's not already
            img = img.convert('RGB')
            # Convert to numpy array for OpenCV operations
            image = np.array(img)

        height, width, _ = image.shape

        # Read the YOLO annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

            # Convert YOLO coordinates to pixel coordinates
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally, add class label
            cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the annotated image
        Image.fromarray(image).save(output_path, 'PNG', icc_profile=None)
        print(f"Saved annotated image to: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


# Example usage
image_dir = r"C:\Users\maxie\Desktop\images\train"
annotation_dir = r"C:\Users\maxie\Desktop\labels\train"
output_dir = r"C:\Users\maxie\Desktop\bbox"

for image_file in os.listdir(image_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpeg"):
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.splitext(image_file)[0] + ".txt"
        annotation_path = os.path.join(annotation_dir, annotation_file)
        output_path = os.path.join(output_dir, f"annotated_{image_file}")

        if os.path.exists(annotation_path):
            draw_yolo_annotations(image_path, annotation_path, output_path)
        else:
            print(f"Annotation file not found for {image_file}")

print("Processing complete.")