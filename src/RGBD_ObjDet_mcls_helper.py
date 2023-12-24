

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from src.TransformerBlock import TransformerBlock


# Modify the model for multi-object detection
class MultiObjectDetectionViT(nn.Module):
    def __init__(self, channels, image_size, embed_size, num_heads, num_classes, max_objects=5, depth_channels=1, dropout=0.5):
        super(MultiObjectDetectionViT, self).__init__()
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, dropout) for _ in range(4)]  # You can adjust the number of blocks
        )
        self.embedding = nn.Linear(channels * image_size * image_size + depth_channels * image_size * image_size, embed_size)
        self.fc_bbox = nn.Linear(embed_size, max_objects * 4)  # Predict bounding boxes for each object
        self.fc_class = nn.Linear(embed_size, max_objects * num_classes)  # Predict class scores for each object
        self.max_objects = max_objects
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, rgb, depth):
        # Flatten and concatenate RGB and depth channels
        x = torch.cat((rgb.view(rgb.size(0), -1), depth.view(depth.size(0), -1)), dim=1)
        x = self.embedding(x)
        x = x.view(x.size(0), 1, -1)  # Add a dimension for the sequence length
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)  # Global average pooling

        # Separate branches for bounding box and class predictions
        bbox_preds = self.fc_bbox(self.dropout(x))
        class_preds = self.fc_class(self.dropout(x))

        # Reshape predictions to (batch_size, max_objects, 4) for bounding boxes
        bbox_preds = bbox_preds.view(-1, self.max_objects, 4)
        # Reshape predictions to (batch_size, max_objects, num_classes) for class scores
        class_preds = class_preds.view(-1, self.max_objects, self.num_classes)

        return bbox_preds, class_preds

# Toy dataset with random RGB, depth images, and bounding boxes for multi-object detection


class RandomMultiObjectDetectionDataset(Dataset):
    def __init__(self, num_samples=100, max_objects=5, image_size=224, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.max_objects = max_objects
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.rgb_data = torch.rand(
            (num_samples, 3, image_size, image_size)).numpy()  # Random RGB images
        self.depth_data = torch.rand(
            (num_samples, 1, image_size, image_size)).numpy()  # Random depth images
        # Random bounding boxes [x_min, y_min, x_max, y_max]
        self.bboxes = torch.randint(
            0, image_size, (num_samples, max_objects, 4)).numpy()
        self.labels = torch.randint(
            0, num_classes, (num_samples, max_objects)).numpy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rgb_image = self.rgb_data[idx]
        depth_image = self.depth_data[idx]
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        rgb_image = self.transform(rgb_image)
        depth_image = self.transform(depth_image)

        return rgb_image, depth_image, bboxes, labels

# Visualize a sample RGB, depth image, and bounding boxes for multi-object detection


def visualize_multi_object_sample(rgb_image, depth_image, bboxes, labels):
    rgb_image = np.transpose(rgb_image.numpy(), (0, 2, 1))
    depth_image = depth_image.numpy().squeeze()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='viridis')
    plt.title("Depth Image")

    plt.figure()
    plt.imshow(rgb_image)

    for bbox, label in zip(bboxes, labels):
        plt.gca().add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none'))
        plt.text(bbox[0], bbox[1], f"Class {label}", color='r')

    plt.title("Bounding Boxes for Multi-Object Detection")
    plt.show()


if __name__ == '__main__':

    # Instantiate the multi-object detection dataset
    multi_object_detection_dataset = RandomMultiObjectDetectionDataset(
        num_samples=100, max_objects=5, image_size=224, num_classes=10, transform=transforms.ToTensor())

    # Visualize a sample
    sample_idx = 0
    rgb_sample, depth_sample, bboxes_sample, labels_sample = multi_object_detection_dataset[
        sample_idx]
    visualize_multi_object_sample(
        rgb_sample, depth_sample, bboxes_sample, labels_sample)

    # Set up the data loader
    dataloader_multi_object = DataLoader(
        multi_object_detection_dataset, batch_size=8, shuffle=True)
