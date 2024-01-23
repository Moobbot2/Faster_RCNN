import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import vision_transformer
# from module import vision_transformer
from config import (
    CLASSES,
    RESIZE_TD,
)
from datasets import train_dataset, train_loader, valid_loader

# Define the Vision Transformer model


class VisionTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_encoder_layers=6):
        super(VisionTransformer, self).__init__()

        self.backbone = vision_transformer.VisionTransformer(
            img_size=(RESIZE_TD[0], RESIZE_TD[1]),
            patch_size=16,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
        )

    def forward(self, x):
        return self.backbone(x)


# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
num_classes = len(CLASSES)
model = VisionTransformer(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs.logits, targets["labels"])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "vision_transformer_model.pth")
