import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import sys
import datasets
import torch
import math
from torchvision import datasets, transforms
from fvcore.nn import FlopCountAnalysis
from IPython.display import display
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import plotly.graph_objects as go
from tqdm import tqdm
from utils import hardware_check, generate_directory_tree, plot_animal_samples_in_single_row, show_sample, show_loss, show_accuracy
from ViT import ViT
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = hardware_check()

# Download Animal Faces Dataset
data_path = kagglehub.dataset_download("andrewmvd/animal-faces")
print("Path to dataset files:", data_path)

# To see the dataset structure, uncomment the following lines:
#tree = generate_directory_tree(data_path)
#print(tree)

# To check some dataset samples, uncomment the following line:
#plot_animal_samples_in_single_row(data_path)

# Dataset and DataLoader creation
transform = transforms.Compose([
    transforms.Resize(config["resize"]),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(config["normalize_mean"], config["normalize_std"])
])

train_dataset = datasets.ImageFolder(root=data_path+"/afhq/train/", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Dataset info
print(f"Found Classes: {train_dataset.classes}")
print(f"Numbers of Training Batches: {len(train_loader)}")

# Dataloader functioning check
#data_iter = iter(train_loader)
#images, labels = next(data_iter)

#print(f"Image Batches Shape: {images.shape}") # [Batch, Channel, H, W]
#print(f"Labels Batch Shape: {labels.shape}")     # [Batch]
#print(f"First label value example: {labels[0].item()}")

# Plot a dataloader sample
#show_sample(images[0], labels[0])

# ViT inizialitazion
model = ViT(
    in_channels=config["in_channels"],
    image_size=config["image_size"],
    patch_size=config["patch_size"],
    number_of_encoder=config["num_encoders"],
    embeddings=config["embedding_dim"],
    d_ff_scale=config["d_ff_scale"],
    heads=config["num_heads"],
    input_dropout_rate=config["input_dropout"],
    attention_dropout_rate=config["attention_dropout"],
    feed_forward_dropout_rate=config["ff_dropout"],
    number_of_classes=config["num_classes"]
).to(device)

print("Model created!")

# Split the train set into train and val
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len

train_subset, val_subset = random_split(train_dataset, [train_len, val_len])

# Create dataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

# Training parameters
num_epochs = config["epochs"]
learning_rate = config["learning_rate"]

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Collect losses for plotting
train_losses = []
val_losses = []

# Collect metrics for plotting
train_metrics = {'ce_loss': [], 'acc': []}
val_metrics = {'ce_loss': [], 'acc': []}

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    epoch_preds = []
    epoch_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        epoch_preds.extend(preds.detach().cpu().numpy())
        epoch_labels.extend(labels.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # to update the model parameters

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    epoch_acc = (np.array(epoch_preds) == np.array(epoch_labels)).mean()

    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    train_metrics['ce_loss'].append(avg_train_loss)
    train_metrics['acc'].append(epoch_acc)

    # validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    val_epoch_preds = []
    val_epoch_labels = []

    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f"Validation"):
            val_images, val_labels = val_batch
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_logits = model(val_images)
            loss = loss_fn(val_logits, val_labels)

            val_preds = torch.argmax(val_logits, dim=1)

            val_epoch_preds.extend(preds.detach().cpu().numpy())
            val_epoch_labels.extend(labels.detach().cpu().numpy())

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = (np.array(val_epoch_preds) == np.array(val_epoch_labels)).mean()
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}\n\n")

    if epoch != 0 and val_acc > max(val_metrics['acc']):
      print('Better val acc reached! Saving the model...')
      torch.save(model.state_dict(), "vit_animal_detection_model_" + str(epoch) + ".pth")
      print("Model parameters saved at epoch ", epoch)


    val_metrics['ce_loss'].append(avg_val_loss)
    val_metrics['acc'].append(val_acc)

print("Training finished!")


# Show loss and accuracy
show_loss(num_epochs, train_metrics, val_metrics)
show_accuracy(num_epochs, train_metrics, val_metrics)

