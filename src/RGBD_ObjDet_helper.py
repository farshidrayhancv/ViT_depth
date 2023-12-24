

import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from src.TransformerBlock import TransformerBlock
from torch.utils.data import Dataset


# Modify the model for object detection
class ObjectDetectionViT(nn.Module):
    def __init__(self, channels, image_size, embed_size, num_heads, num_classes, depth_channels=1, num_boxes=4, dropout=0.5):
        super(ObjectDetectionViT, self).__init__()
        self.transformer_blocks = nn.Sequential(
            # You can adjust the number of blocks
            *[TransformerBlock(embed_size, num_heads, dropout) for _ in range(4)]
        )
        self.embedding = nn.Linear(
            channels * image_size * image_size + depth_channels * image_size * image_size, embed_size)
        self.fc_bbox = nn.Linear(embed_size, num_boxes)
        self.fc_class = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rgb, depth):
        # Flatten and concatenate RGB and depth channels
        x = torch.cat((rgb.view(rgb.size(0), -1),
                      depth.view(depth.size(0), -1)), dim=1)
        x = self.embedding(x)
        x = x.view(x.size(0), 1, -1)  # Add a dimension for the sequence length
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)  # Global average pooling

        # Separate branches for bounding box and class predictions
        bbox_preds = self.fc_bbox(self.dropout(x))
        class_preds = self.fc_class(self.dropout(x))

        return bbox_preds, class_preds

# Toy dataset with random RGB, depth images, and bounding boxes


class RandomObjectDetectionDataset(Dataset):
    def __init__(self, num_samples=100, image_size=224, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.rgb_data = torch.rand(
            (num_samples, 3, image_size, image_size)).numpy()  # Random RGB images
        self.depth_data = torch.rand(
            (num_samples, 1, image_size, image_size)).numpy()  # Random depth images
        # Random bounding boxes [x_min, y_min, x_max, y_max]
        self.bboxes = torch.randint(0, image_size, (num_samples, 4))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rgb_image = self.rgb_data[idx]
        depth_image = self.depth_data[idx]
        bbox = self.bboxes[idx]
        label = self.labels[idx]

        rgb_image = self.transform(rgb_image)
        depth_image = self.transform(depth_image)

        return rgb_image, depth_image, bbox, label

# Visualize a sample RGB, depth image, and bounding box


def visualize_sample(rgb_image, depth_image, bbox):
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
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0],
                                      bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title("Bounding Box")

    plt.show()


if __name__ == '__main__':

    # Instantiate the toy dataset
    object_detection_dataset = RandomObjectDetectionDataset(
        num_samples=100, image_size=224, num_classes=10, transform=transforms.ToTensor())

    # Visualize a sample
    sample_idx = 5
    rgb_sample, depth_sample, bbox_sample, label_sample = object_detection_dataset[
        sample_idx]
    visualize_sample(rgb_sample, depth_sample, bbox_sample)

    # Instantiate the model and move it to the device
    object_detection_model = ObjectDetectionViT(
        channels=3, image_size=224, embed_size=256, num_heads=8, num_classes=10, depth_channels=1)
