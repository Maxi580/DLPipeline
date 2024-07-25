import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import os


def mix_images(image1, image2, alpha=0.5):
    # Ensure both images have the same size and mode
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.LANCZOS)
    if image1.mode != image2.mode:
        image2 = image2.convert(image1.mode)

    return Image.blend(image1, image2, alpha)
