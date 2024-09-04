import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from utils import *

MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')
MODEL_INFERENCE_UNET_OUTPUT_DIR = os.getenv('MODEL_INFERENCE_UNET_OUTPUT_DIR')
UNET_NUM_CLASSES = os.getenv('UNET_NUM_CLASSES')


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


def load_unet_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            model = state_dict['model']
        else:
            model = UNet(n_channels=3, n_classes=NUM_CLASSES)
            model.load_state_dict(state_dict['model_state_dict'])
        return model.eval()
    else:
        raise ValueError("Unexpected format of saved model")


def run_inference(model, image):
    model.eval()
    with torch.no_grad():
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            model = model.cuda()
        output = model(image_tensor)
        return output.squeeze(0).cpu()


def process_results(output, original_image):
    # Convert the output to a numpy array and apply argmax to get the predicted class for each pixel
    prediction = output.argmax(dim=0).numpy()

    # Create a color map for visualization (adjust colors as needed)
    color_map = np.array([[0, 0, 0],  # Background
                          [255, 0, 0],  # Class 1
                          [0, 255, 0],  # Class 2
                          [0, 0, 255]])  # Class 3 (add more colors if you have more classes)

    # Apply the color map to the prediction
    segmentation_map = color_map[prediction]

    # Blend the segmentation map with the original image
    alpha = 0.5  # Adjust this value to change the blend ratio
    blended = cv2.addWeighted(original_image, 1 - alpha, segmentation_map, alpha, 0)

    return blended


def unet_inference():
    models = []
    for filename in os.listdir(MODEL_INFERENCE_UNET_OUTPUT_DIR):
        if filename.endswith('.pth'):
            model_path = os.path.join(MODEL_INFERENCE_UNET_OUTPUT_DIR, filename)
            try:
                model = load_unet_model(model_path)
                models.append((filename, model))
                print(f"Loaded model: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    for image_filename in os.listdir(MODEL_INFERENCE_INPUT):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(MODEL_INFERENCE_INPUT, image_filename)

            image = Image.open(input_path).convert('RGB')
            original_image = np.array(image)

            for model_name, model in models:
                output = run_inference(model, image)

                segmentation_result = process_results(output, original_image)

                base_name, ext = os.path.splitext(image_filename)
                output_filename = f"{base_name}_{ext}"
                output_path = os.path.join(MODEL_INFERENCE_UNET_OUTPUT_DIR, model_name, output_filename)

                cv2.imwrite(output_path, cv2.cvtColor(segmentation_result, cv2.COLOR_RGB2BGR))
                print(f"Processed {image_filename} with {model_name}")


def unet_main():
    does_exist = check_directory_content([MODEL_INFERENCE_INPUT, MODEL_INFERENCE_UNET_OUTPUT_DIR])

    if does_exist:
        unet_inference()
    else:
        print("No U-net Model has been found.")