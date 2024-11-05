# Step 1: Configure Git with your GitHub identity
!git config --global user.email "micahwinget38@gmail.com"
!git config --global user.name "mwing-dev"

# Step 2: Clone the private GitHub repository using the token
token = "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"  # Replace with your GitHub PAT
username = "mwing-dev"
repo_name = "Reload"

# Clone the private repository using the PAT
!git clone https://{username}:{token}@github.com/{username}/{repo_name}.git
repo_path = f'/content/{repo_name}'

# Step 3: Import Required Libraries
!pip install gradio  # Install Gradio
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Step 4: Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (128 // 4) * (128 // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 5: Initialize Model and Load Pre-trained Weights
num_classes = 3  # Adjust this based on your number of classes
model = SimpleCNN(num_classes=num_classes)

# Load the trained model weights from the cloned repository
model_path = os.path.join(repo_path, "/content/Reload/Code/Training/training_model.pth")  # Replace with the correct path inside the repo
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Step 6: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class names (ensure they match the training class order)
class_names = ['No_Case', 'No_Powder', 'Yes_Powder']

# Step 7: Gradio Function to Process Uploaded Image and Display Result
def classify_image(image):
    # Prepare the image for model prediction
    image = image.convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Run the model prediction
    with torch.no_grad():
        output = model(transformed_image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get prediction details
    predicted_class = class_names[predicted.item()]
    confidence_percentage = confidence.item() * 100

    # Return the prediction result without saving the image
    return {
        "Prediction": predicted_class,
        "Confidence": f"{confidence_percentage:.2f}%"
    }

# Step 8: Create Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Image Classification - Pass/Fail"
)

# Step 9: Launch Gradio Interface
interface.launch()
