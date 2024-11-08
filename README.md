# Image Classification - Pass/Fail with Raspberry Pi Pico

This project uses an image classification model to detect the presence of a specific substance (e.g., powder) in an image, returning a "Pass" or "Fail" result. The classification result is sent to a Raspberry Pi Pico microcontroller, which lights up specific LEDs to indicate the result.

## Features

- **Image Classification**: A CNN model classifies images into "Yes_Powder" or "No_Powder".
- **Real-Time Feedback**: Based on the classification result, the system sends commands to a Raspberry Pi Pico via USB.
- **LED Indicators**: The Pico activates a "Pass" LED if powder is detected or a "Fail" LED otherwise.

## Project Components

- **Dockerized Python Application**: Runs the Gradio-based web interface and communicates with the Pico.
- **Raspberry Pi Pico**: Controls LEDs based on received commands.
- **Gradio Interface**: Web interface to upload images and view classification results.

## Requirements

### Hardware
- Raspberry Pi Pico with CircuitPython installed.
- Two LEDs connected to GPIO pins 14 (Fail) and 15 (Pass) on the Pico.
- USB connection between the Pico and the host machine.

### Software
- Docker and Docker Compose installed on the host machine.
- Python packages (see `requirements.txt`):
  - `torch`, `torchvision`, `Pillow`, `gradio`, `pyserial`
- Model weights in a file named `Trained_Model.pth` located in `/app/Reload_Trained_Model`.

## Setup and Installation

### 1. Set Up Raspberry Pi Pico

1. Install **CircuitPython** on the Pico.
2. Copy the following code to `boot.py` on the Pico to enable USB data communication:

   ```python
   import usb_cdc
   usb_cdc.enable(data=True)
