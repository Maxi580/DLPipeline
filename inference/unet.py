import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import logging
import gc

from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_UNET_OUTPUT_DIR = os.getenv('MODEL_INFERENCE_UNET_OUTPUT_DIR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)
        logits = self.outc(x)
        return logits


def get_optimal_dimensions(width, height):
    """Images need to be divisible by 2^n"""
    new_width = ((width + 15) // 16) * 16
    new_height = ((height + 15) // 16) * 16
    return new_width, new_height


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_color_map(num_classes):
    np.random.seed(42)  # for reproducibility
    return np.array([np.random.randint(0, 256, 3) for _ in range(num_classes)])


def load_unet_model(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = UNet(n_channels=checkpoint['n_channels'], n_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model
    except Exception as e:
        logger.error(f"Failed to load unet model from {model_path}: {e}")
        return None


def run_inference(model, image):
    model.eval()
    with torch.no_grad():
        original_height, original_width = image.shape[:2]

        new_width, new_height = get_optimal_dimensions(original_width, original_height)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(device)
        output = model(image_tensor)

        output_resized = F.interpolate(output, size=(original_height, original_width), mode='bilinear',
                                       align_corners=False)

        return output_resized.squeeze(0).cpu()


def process_results(output, original_image, color_map):
    prediction = output.argmax(dim=0).numpy()
    segmentation_map = color_map[prediction]

    alpha = 0.5  # Adjust this value to change the blend ratio
    blended = cv2.addWeighted(original_image, 1 - alpha, segmentation_map.astype(np.uint8), alpha, 0)

    return blended


def unet_inference():
    models = []
    for filename in os.listdir(MODEL_INFERENCE_UNET_OUTPUT_DIR):
        if filename.endswith('.pth'):
            model_path = os.path.join(MODEL_INFERENCE_UNET_OUTPUT_DIR, filename)
            try:
                model = load_unet_model(model_path)
                if model is not None:
                    num_classes = model.n_classes
                    color_map = get_color_map(num_classes)
                    models.append((filename, model, color_map))
                    logger.info(f"Loaded model: {filename} with {num_classes} classes")
                else:
                    logger.warning(f"Failed to load model: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")

    for image_filename in os.listdir(MODEL_INFERENCE_INPUT):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, image_filename)
            original_image = cv2.imread(input_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            for model_name, model, color_map in models:
                output = run_inference(model, original_image)
                segmentation_result = process_results(output, original_image, color_map)

                model_output_dir = os.path.join(MODEL_INFERENCE_UNET_OUTPUT_DIR, os.path.splitext(model_name)[0])
                os.makedirs(model_output_dir, exist_ok=True)

                output_path = os.path.join(model_output_dir, image_filename)

                cv2.imwrite(output_path, cv2.cvtColor(segmentation_result, cv2.COLOR_RGB2BGR))
                logger.info(f"Processed {image_filename} with {model_name}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def unet_main():
    if check_directory_content([MODEL_INFERENCE_UNET_OUTPUT_DIR]):
        print("Starting unet inference")
        unet_inference()
    else:
        print("No U-Net models found in the specified directory.")
