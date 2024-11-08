import os
import serial
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time

# Define the CNN model
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

# Initialize the model and load trained weights
num_classes = 3
model = SimpleCNN(num_classes=num_classes)
model_path = os.path.join('/app/Reload_Trained_Model', "Trained_Model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set device for the model (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define class names
class_names = ['No_Case', 'No_Powder', 'Yes_Powder']

# Check if the serial device exists and connect if available
pico_serial = None
serial_port = '/dev/ttyACM1'  # Corrected the serial port to ttyACM1
if os.path.exists(serial_port):
    pico_serial = serial.Serial(serial_port, 115200)
    print(f"Connected to {serial_port}")
else:
    print(f"Warning: {serial_port} not found. Running without serial connection.")

# Function to classify the image and optionally send results over serial
def classify_image(image):
    # Preprocess the image
    image = image.convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(transformed_image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Extract prediction results
    predicted_class = class_names[predicted.item()]
    confidence_percentage = confidence.item() * 100
    result = "Pass" if predicted_class == "Yes_Powder" else "Fail"

    # Send result over serial if the device is connected
    if pico_serial:
        # Send "pass" or "fail" command to the Pico
        command = "pass" if result == "Pass" else "fail"
        pico_serial.write(f"{command}\n".encode())
        print(f"Sent '{command}' to Pico")
    else:
        print("Serial device not available, skipping serial communication.")

    # Return prediction results
    return {
        "Prediction": predicted_class,
        "Confidence": f"{confidence_percentage:.2f}%",
        "Result": result
    }

# Set up the Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Image Classification - Pass/Fail"
)

# Launch the Gradio app
interface.launch(server_name="0.0.0.0", server_port=7860)

# Close serial connection if it was opened
if pico_serial:
    pico_serial.close()

