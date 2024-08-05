import os
from PIL import Image
from torchvision.ops import box_iou
from collections import defaultdict
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn,
                                          fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
import torch.optim as optim

FRCNN_MODELS = os.getenv('FRCNN_MODELS').split(',')
PRETRAINED = bool(os.getenv('PRETRAINED'))
FRCNN_EPOCHS = int(os.getenv('FRCNN_EPOCHS'))
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR')
FRCNN_BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

OPTIMIZER_LEARNING_RATE = float(os.getenv('OPTIMIZER_LEARNING_RATE'))
OPTIMIZER_MOMENTUM = float(os.getenv('OPTIMIZER_MOMENTUM'))
OPTIMIZER_WEIGHT_DECAY = float(os.getenv('OPTIMIZER_WEIGHT_DECAY'))
SCHEDULER_STEP_SIZE = int(os.getenv('SCHEDULER_STEP_SIZE'))
SCHEDULER_GAMMA = float(os.getenv('SCHEDULER_GAMMA'))
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE'))
EARLY_STOPPING_MIN_DELTA = float(os.getenv('EARLY_STOPPING_MIN_DELTA'))

AVAILABLE_MODELS = {
    'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    'fasterrcnn_mobilenet_v3_large_320_fpn': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    'fasterrcnn_resnet50_fpn_v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    'ssd300_vgg16': ssd300_vgg16,
    'retinanet_resnet50_fpn_v2': retinanet_resnet50_fpn_v2,
}


class CustomDataset(Dataset):
    """Loads Data (Converts Yolo txt annotations to x_min, y_min, x_max, y_max)"""

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        if len(boxes) > 0:
            h, w = image.shape[-2:]
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]),
                  "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                  "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)}

        return image, target


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_model(model_name, num_classes, pretrained=True):
    """
    Load a model and modify it for the given number of classes.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from {list(AVAILABLE_MODELS.keys())}")

    if pretrained:
        model = AVAILABLE_MODELS[model_name](weights='DEFAULT')

        # represents the number of channels (or depth) of the feature map for each region of interest (RoI) after
        # it has gone through the backbone network and RoI pooling/alignment.
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Creating a new fully connectedLayers: Replace in_features (input) and num_classes(ouput)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = AVAILABLE_MODELS[model_name](weights=None, num_classes=num_classes)

    return model


def get_num_classes(label_dir):
    class_ids = set()
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
    return len(class_ids)


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = torch.sum(torch.stack(list(loss_dict.values())))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    average_loss = total_loss / num_batches
    return average_loss


def calculate_average_precision(scores, correct):
    sorted_indices = torch.argsort(scores, descending=True)
    correct = correct[sorted_indices]
    precisions = torch.cumsum(correct, dim=0) / torch.arange(1, len(correct) + 1)
    average_precision = torch.sum(precisions * correct) / max(1, correct.sum())
    return average_precision.item()


def calculate_mAP(all_predictions, all_targets, iou_threshold):
    ap_per_class = defaultdict(list)

    for (pred_boxes, pred_scores, pred_labels), (target_boxes, target_labels) in zip(all_predictions, all_targets):
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue

        ious = box_iou(pred_boxes, target_boxes)

        for class_id in torch.unique(target_labels):
            class_pred_indices = torch.tensor(pred_labels == class_id).nonzero().squeeze(1)
            class_target_indices = torch.tensor(target_labels == class_id).nonzero().squeeze(1)

            if len(class_pred_indices) == 0 or len(class_target_indices) == 0:
                continue

            class_pred_scores = pred_scores[class_pred_indices]
            class_ious = ious[class_pred_indices][:, class_target_indices]

            max_ious, max_indices = class_ious.max(dim=1)
            correct = max_ious > iou_threshold

            ap = calculate_average_precision(class_pred_scores, correct)
            ap_per_class[class_id.item()].append(ap)

    mAP = sum(sum(aps) / len(aps) for aps in ap_per_class.values()) / len(ap_per_class)
    return mAP


@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()

            target_boxes = target['boxes'].to(device)
            target_labels = target['labels'].to(device)

            all_predictions.append((pred_boxes, pred_scores, pred_labels))
            all_targets.append((target_boxes, target_labels))

    return calculate_mAP(all_predictions, all_targets, iou_threshold)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_dataset(train_image_dir, train_label_dir, val_image_dir, val_label_dir):
    transform = T.Compose([
        T.ToTensor(),
    ])

    train_dataset = CustomDataset(image_dir=train_image_dir,
                                  label_dir=train_label_dir,
                                  transform=transform)
    val_dataset = CustomDataset(image_dir=val_image_dir,
                                label_dir=val_label_dir,
                                transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FRCNN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FRCNN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def save_model(model, epoch, optimizer, loss, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Model saved to {save_path}")


def create_faster_rcnn_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, name):
    train_loader, val_loader = load_dataset(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        val_image_dir=val_image_dir,
        val_label_dir=val_label_dir
    )
    for rcnn_model in FRCNN_MODELS:
        model = load_model(rcnn_model, get_num_classes(train_label_dir), PRETRAINED)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=OPTIMIZER_LEARNING_RATE, momentum=OPTIMIZER_MOMENTUM,
                              weight_decay=OPTIMIZER_WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
        num_epochs = FRCNN_EPOCHS
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

        print(f"Starting training for {rcnn_model}")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Val loader length: {len(val_loader)}")
        print(f"Device: {device}")
        for epoch in range(num_epochs):
            loss = train_one_epoch(model, optimizer, train_loader, device)

            mAP = evaluate(model, val_loader, device)
            scheduler.step()

            output_dir = os.path.join(MODEL_OUTPUT_DIR, f"{name}_{model}")
            save_model(model, epoch, optimizer, loss, output_dir)
            print(f"Epoch {epoch + 1}, Loss: {loss}, Validation mAP: {mAP}")

            early_stopping(mAP)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print(f"Faster RCNN {model} has finished")

