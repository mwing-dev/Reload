# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import Required Libraries
!pip install gradio  # Install Gradio
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime

# Step 3: Define the CNN Model
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

# Step 4: Initialize Model and Load Pre-trained Weights
num_classes = 3 
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load('/content/drive/MyDrive/Pytorch/training_model.pth'))
model.eval()

# Step 5: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 6: Set Paths for Output Subfolders
output_folder = '/content/drive/MyDrive/Pytorch/Predictions'
pass_folder = os.path.join(output_folder, "Pass")
fail_folder = os.path.join(output_folder, "Fail")

# Create the output folders if they don't exist
os.makedirs(pass_folder, exist_ok=True)
os.makedirs(fail_folder, exist_ok=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class names (ensure they match the training class order)
class_names = ['No_Case', 'No_Powder', 'Yes_Powder']

# Step 7: Gradio Function to Process Uploaded Image and Display Result
def classify_and_save(image):
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

    # Generate a timestamp and construct the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_name = f"{predicted_class}_{confidence_percentage:.2f}_{timestamp}.jpg"

    # Determine the output path based on the predicted class
    if predicted_class == "Yes_Powder":
        output_image_path = os.path.join(pass_folder, output_image_name)
        result = "Pass"
    else:
        output_image_path = os.path.join(fail_folder, output_image_name)
        result = "Fail"

    # Save the image to the appropriate folder
    image.save(output_image_path)

    # Return result for display
    return {
        "Prediction": predicted_class,
        "Confidence": f"{confidence_percentage:.2f}%",
        "Result": result
    }

# Step 8: Create Gradio Interface with updated syntax
interface = gr.Interface(
    fn=classify_and_save,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Image Classification - Pass/Fail"
)

# Step 9: Launch Gradio Interface
interface.launch()