import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

UNET_BATCH_SIZE = int(os.getenv('UNET_BATCH_SIZE'))
UNET_NUM_CLASSES = int(os.getenv('UNET_NUM_CLASSES'))
UNET_NUM_WORKERS = int(os.getenv('UNET_NUM_WORKERS'))
UNET_EPOCHS = int(os.getenv('UNET_EPOCHS'))
UNET_LR = float(os.getenv('UNET_LR'))

MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')


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


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir)])
        self.mask_files = sorted([f for f in os.listdir(mask_dir)])
        self.transform = transforms.ToTensor()

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

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask


def save_model(model, epoch, optimizer, val_loss, name):
    output_dir = os.path.join(MODEL_OUTPUT_DIR, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{epoch}_{name}.pth")
    torch.save({
        'epoch': epoch,
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, output_path)
    print(f"Model saved to {output_path}")


def train_unet(model, train_loader, val_loader, name, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Found device: {device}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        print(f"Starting with epoch: {epoch}")
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1).long())
                val_loss += loss.item()

        scheduler.step(val_loss)

        print(f"Learning Rate: {scheduler.get_last_lr()}")

        save_model(model, epoch, optimizer, val_loss, name)
        print(f"Model {epoch} saved with validation loss: {val_loss:.4f} and training loss: {train_loss:.4f}")

    return model


def create_u_net_model(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, name):
    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir)
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=UNET_BATCH_SIZE, shuffle=True, num_workers=UNET_NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=UNET_BATCH_SIZE, shuffle=False, num_workers=UNET_NUM_WORKERS)

    model = UNet(n_channels=3, n_classes=UNET_NUM_CLASSES)
    train_unet(model, train_loader, val_loader, name, UNET_EPOCHS, UNET_LR)
