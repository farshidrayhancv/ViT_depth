
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from RGBD_classification_helper import RGBDViT, RandomRGBDDataset, visualize_sample


class RandomRGBDDataset(Dataset):
    def __init__(self, num_samples=100, image_size=224, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.rgb_data = torch.rand(
            (num_samples, 3, image_size, image_size)).numpy()  # Random RGB images
        self.depth_data = torch.rand(
            (num_samples, 1, image_size, image_size)).numpy()  # Random depth images
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rgb_image = self.rgb_data[idx]
        depth_image = self.depth_data[idx]
        label = self.labels[idx]

        rgb_image = self.transform(rgb_image)
        depth_image = self.transform(depth_image)

        return rgb_image, depth_image, label


# Instantiate the toy dataset
toy_dataset = RandomRGBDDataset(
    num_samples=100, image_size=224, num_classes=10, transform=transforms.ToTensor())
dataloader = DataLoader(toy_dataset, batch_size=8, shuffle=True)


# Evaluation loop


# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the device
model = RGBDViT(channels=3, image_size=224, embed_size=256,
                num_heads=8, num_classes=10, depth_channels=1)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for rgb, depth, labels in dataloader:
        rgb, depth, labels = rgb.to(device), depth.to(
            device), labels.to(device)

        # Forward pass
        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

print("Training complete.")
