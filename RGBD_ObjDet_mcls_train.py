
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from RGBD_ObjDet_mcls_helper import MultiObjectDetectionViT, RandomMultiObjectDetectionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

multi_object_detection_dataset = RandomMultiObjectDetectionDataset(num_samples=100, max_objects=5, image_size=224, num_classes=10, transform=transforms.ToTensor())
dataloader_multi_object = DataLoader(multi_object_detection_dataset, batch_size=8, shuffle=True)

# Instantiate the model and move it to the device
multi_object_detection_model = MultiObjectDetectionViT(channels=3, image_size=224, embed_size=256, num_heads=8, num_classes=10, max_objects=5, depth_channels=1)
multi_object_detection_model.to(device)

# Define loss function and optimizer
criterion_bbox = nn.MSELoss()  # Mean Squared Error loss for bounding box predictions
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(multi_object_detection_model.parameters(), lr=0.001)

# Set parameters for bounding box and class predictions
max_objects = 5
num_classes = 10

# Training loop for multi-object detection
num_epochs = 10
for epoch in range(num_epochs):
    multi_object_detection_model.train()
    running_loss_bbox = 0.0
    running_loss_class = 0.0

    for rgb, depth, bboxes, labels in dataloader_multi_object:
        rgb, depth, bboxes, labels = rgb.to(device), depth.to(device), bboxes.to(device), labels.to(device)

        bboxes = bboxes.float()
        # Forward pass
        bbox_preds, class_preds = multi_object_detection_model(rgb, depth)

        # Flatten ground truth for loss computation
        bboxes_flat = bboxes.view(-1, 4)
        labels_flat = labels.view(-1)

        # Flatten predictions for loss computation
        bbox_preds_flat = bbox_preds.view(-1, 4)
        class_preds_flat = class_preds.view(-1, num_classes)

        # Compute losses
        loss_bbox = criterion_bbox(bbox_preds_flat, bboxes_flat)
        loss_class = criterion_class(class_preds_flat, labels_flat)

        # Total loss
        loss = loss_bbox + loss_class

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss_bbox += loss_bbox.item()
        running_loss_class += loss_class.item()

    # Print average loss for the epoch
    average_loss_bbox = running_loss_bbox / len(dataloader_multi_object)
    average_loss_class = running_loss_class / len(dataloader_multi_object)
    print(f"Epoch {epoch + 1}/{num_epochs}, Bbox Loss: {average_loss_bbox:.4f}, Class Loss: {average_loss_class:.4f}")

print("Training complete.")
