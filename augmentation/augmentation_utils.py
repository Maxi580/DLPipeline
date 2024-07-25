import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import os


def flip_horizontal(image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def rotate(image):
    angle = random.randint(-45, 45)
    rotated = image.rotate(angle, expand=True)
    return rotated.resize(image.size, Image.LANCZOS)


def adjust_brightness(image):
    factor = random.uniform(0.5, 4)
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image):
    factor = random.uniform(0.5, 2.5)
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_saturation(image):
    factor = random.uniform(0.5, 1.5)
    return ImageEnhance.Color(image).enhance(factor)


def add_noise(image):
    img_array = np.array(image)
    noise = np.random.normal(0, 25, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))


def sharpen(image):
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


