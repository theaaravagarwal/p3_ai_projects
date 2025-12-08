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
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -----------------------------
# 1. LOAD DATA
# Transforms are preprocessing steps that get applied automatically to every image
# you load from a dataset. 
# Think of transforms as a recipe that says:
#
# “Every time you give me an image, do X, then Y, then Z to it.”
# “For every MNIST image: convert it to a PyTorch tensor.
# MNIST images come in as PIL images (Python Imaging Library).
#
# But your neural network expects tensors.
# -----------------------------
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load train dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform_train,
    download=True
)

# Load test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform_test,
    download=True
)

# Make DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

unique_labels = sorted(set(train_dataset.targets.tolist()))
print('Unique labels in training dataset:', unique_labels)

# -----------------------------
# 2. DEFINE NEURAL NETWORK
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.norm(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleNN().to(device)
print(model)

# -----------------------------
# 3. LOSS FUNCTION + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# -----------------------------
# 4. TRAINING LOOP
# -----------------------------

epochs = 35
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # ---- Save checkpoint after each epoch ----
    checkpoint_path = f"mnist_simplenn_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# -----------------------------
# 5. EVALUATION
# -----------------------------
correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ---- Save final model after evaluation ----
final_model_path = "mnist_simplenn_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model: {final_model_path}")

# -----------------------------
# 6. TEST SINGLE PREDICTION
# -----------------------------
# Gradio sketchpad returns a full-color NumPy array (H,W,3).
# MNIST images are grayscale (1x28x28) and normalized.
# This preprocessing converts user drawings into MNIST format.
# -----------------------------

# MNIST normalization values:
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

def preprocess_image(image):
    """Convert Gradio Sketchpad output to a normalized 1x28x28 tensor."""

    # Gradio may pass {'composite': array}
    if isinstance(image, dict) and "composite" in image:
        image = image["composite"]

    # Define preprocessing pipeline
    sketch_transform = transforms.Compose([
        transforms.ToPILImage(),                      # NumPy → PIL
        transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
        transforms.Resize((28, 28)),                  # Match MNIST input
        transforms.Lambda(lambda img: ImageOps.invert(img)),
        transforms.ToTensor(),                        # → (1, 28, 28), values in [0,1]
        transforms.Normalize(MNIST_MEAN, MNIST_STD),  # Match MNIST training normalization
    ])

    tensor = sketch_transform(image)                  # Shape: (1, 28, 28)
    tensor = tensor.unsqueeze(0)                      # Shape: (1, 1, 28, 28)
    return tensor.to(device)                          # Move to same device as model


def predict_digit(image):
    """Take raw Sketchpad input → return predicted digit + confidence."""

    if image is None:
        return "Draw something!"

    # Convert to model input format
    input_tensor = preprocess_image(image)

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)

        # turn logits into probabilities
        probs = torch.softmax(logits, dim=1)

        # predicted class index
        predicted_class = torch.argmax(probs, dim=1).item()

        # confidence of that class
        confidence = probs[0, predicted_class].item()

    # return nicely formatted output
    return f"{predicted_class}  ({confidence * 100:.2f}% confidence)"


# -----------------------------
# GRADIO UI
# -----------------------------
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw Here"),
    outputs="label",
    live=False,
)

interface.queue().launch()