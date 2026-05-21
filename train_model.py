import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------
# Dataset
# -----------------------
dataset = datasets.ImageFolder("datasets", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Classes:", dataset.classes)

# -----------------------
# Model (EfficientNet)
# -----------------------
model = create_model("efficientnet_b0", pretrained=True, num_classes=2)
model = model.to(device)

# -----------------------
# Loss & optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# -----------------------
# Training loop
# -----------------------
epochs = 3

for epoch in range(epochs):
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# -----------------------
# Save model
# -----------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/deepfake_model.pth")

print("✅ Model saved successfully!")