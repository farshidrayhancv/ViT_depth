from RGBD_ObjDet_helper import ObjectDetectionViT, RandomObjectDetectionDataset 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

object_detection_dataset = RandomObjectDetectionDataset(num_samples=100, image_size=224, num_classes=10, transform=transforms.ToTensor())

# Set up the data loader
dataloader = DataLoader(object_detection_dataset, batch_size=8, shuffle=True)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Instantiate the model and move it to the device
object_detection_model = ObjectDetectionViT(channels=3, image_size=224, embed_size=256, num_heads=8, num_classes=10, depth_channels=1)
object_detection_model.to(device)

# Define loss function and optimizer
criterion_bbox = nn.MSELoss()  # Mean Squared Error loss for bounding box predictions
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(object_detection_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    object_detection_model.train()
    running_loss_bbox = 0.0
    running_loss_class = 0.0

    for rgb, depth, bbox, labels in dataloader:
        rgb, depth, bbox, labels = rgb.to(device), depth.to(device), bbox.to(device), labels.to(device)
        
        bbox = bbox.float()

        # Forward pass
        bbox_preds, class_preds = object_detection_model(rgb, depth)
        loss_bbox = criterion_bbox(bbox_preds, bbox)
        
        # print(typeclass_preds)
        loss_class = criterion_class(class_preds, labels)
        
        

        # Total loss
        loss = loss_bbox + loss_class

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss_bbox += loss_bbox.item()
        running_loss_class += loss_class.item()

    # Print average loss for the epoch
    average_loss_bbox = running_loss_bbox / len(dataloader)
    average_loss_class = running_loss_class / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Bbox Loss: {average_loss_bbox:.4f}, Class Loss: {average_loss_class:.4f}")

print("Training complete.")