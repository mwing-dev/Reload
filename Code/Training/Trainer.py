# Step 1: Clone GitHub Repository with authentication
!git config --global user.email "micahwinget38@gmail.com"
!git config --global user.name "mwing-dev"

# Clone the repository (replace this with your actual token, username, and repo URL)
!git clone https://github.com/mwing-dev/Reload.git
repo_path = '/content/Reload'

# Define the path for the Training directory in your repository
training_dir = os.path.join(repo_path, 'Code/Training')

# Step 2: Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define Image Size
image_size = 128

# Step 3: Set Up Data Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 4: Load the Dataset
# Using the Training_images directory from the cloned GitHub repo
data_dir = os.path.join(repo_path, 'Training_images')
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Step 5: Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 6: Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
num_classes = len(dataset.classes)  # Get the number of classes from the dataset
model = SimpleCNN(num_classes=num_classes)

# Step 7: Set Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 9: Training Loop
num_epochs = 20  # Number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

        # Optional: Print batch progress (especially useful for long training)
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Batch Loss: {loss.item():.4f}")

    # Print epoch loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Optional: Save model checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(training_dir, f'training_model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Add, commit, and push the checkpoint to GitHub
        !git -C {repo_path} add .
        !git -C {repo_path} commit -m "Checkpoint for epoch {epoch + 1}"
        !git -C {repo_path} push origin main

# Final model save after training
model_save_path = os.path.join(training_dir, 'training_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Commit and push the final model to GitHub
!git -C {repo_path} add .
!git -C {repo_path} commit -m "Final model after training"
!git -C {repo_path} push origin main
