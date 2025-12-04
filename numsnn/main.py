# -----------------------------
# MNIST DIGIT CLASSIFIER (PyTorch)
# -----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. LOAD DATA (Optimized)
# -----------------------------
# Upgraded transforms: slightly stronger augmentation + efficient normalization
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load datasets
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_train, download=True)
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform_test, download=True)

# DataLoaders (pin_memory improves GPU transfer speed)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

unique_labels = sorted(set(train_dataset.targets.tolist()))
print("Unique labels in training dataset:", unique_labels)


# -----------------------------
# 2. DEFINE OPTIMIZED NEURAL NETWORK
# -----------------------------
# Same structure as yours but cleaner, faster, better activation flow,
# and using nn.Sequential (recommended in modern PyTorch)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                     
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.LayerNorm(128),                # stabilizes learning
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Move model to GPU if available
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
print(f"Using device: {device}")


# -----------------------------
# 3. LOSS FUNCTION + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()

# Adam is good, but add weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Optional but improves training speed on GPU
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


# -----------------------------
# 4. TRAINING LOOP (Optimized)
# -----------------------------
epochs = 25  # trains faster, needs fewer epochs due to better architecture

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training (2–3x faster on Colab GPU)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")


# -----------------------------
# 5. EVALUATION (Optimized + Accurate)
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# -----------------------------
# 6. TEST SINGLE PREDICTION (Gradio)
# -----------------------------
def preprocess_image(image):
    # Preprocessing pipeline for Sketchpad drawings
    sketch_transform = transforms.Compose([
        transforms.ToPILImage(),              
        transforms.Grayscale(),               
        transforms.Resize((28, 28)),         
        transforms.Lambda(lambda img: ImageOps.invert(img)),  
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if isinstance(image, dict):
        image = image["composite"]

    img_tensor = sketch_transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

    return img_tensor.to(device)


def predict_digit(image):
    if image is None:
        return "Draw something!"

    img_tensor = preprocess_image(image)

    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_digit = torch.argmax(prediction).item()

    return str(predicted_digit)


interface = gr.Interface(fn=predict_digit, inputs=gr.Sketchpad(label="Draw Here"), outputs="label")
interface.queue().launch()
