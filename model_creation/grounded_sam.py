import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
import numpy as np
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir)])
        self.mask_files = sorted([f for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img_basename = os.path.splitext(os.path.basename(img_name))[0]
        mask_name = next((f for f in self.mask_files if os.path.splitext(f)[0] == img_basename), None)

        if mask_name is None:
            raise FileNotFoundError(f"No matching mask found for image: {img_name}")

        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        return image, mask


class SAMFineTuner(nn.Module):
    def __init__(self, sam_checkpoint):
        super().__init__()
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.train()  # Set SAM to training mode

    def forward(self, image, points):
        return self.sam(image, points)


def fine_tune_sam(sam_checkpoint, train_dataset, val_dataset, name, num_epochs=10, batch_size=1, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SAMFineTuner(sam_checkpoint).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            # Train Phase: Predict masks and adjust model upon result
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), name + '.pth')
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    print("Fine-tuning completed!")


def create_grounded_sam_model(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, name):
    train_dataset = CustomDataset(train_image_dir, train_mask_dir)
    val_dataset = CustomDataset(val_image_dir, val_mask_dir)

    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Pre-Trained Model

    fine_tune_sam(sam_checkpoint, train_dataset, val_dataset, name, EPOCHS, BATCH_SIZE, LEARNING_RATE)
