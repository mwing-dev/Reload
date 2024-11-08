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
   
3. Copy the main code to `code.py` on the Pico:
   
   ```python
   import time
   import digitalio
   import board
   import usb_cdc

   pass_led = digitalio.DigitalInOut(board.GP15)
   pass_led.direction = digitalio.Direction.OUTPUT
   fail_led = digitalio.DigitalInOut(board.GP14)
   fail_led.direction = digitalio.Direction.OUTPUT

   usb_serial = usb_cdc.data

   while True:
       if usb_serial and usb_serial.in_waiting > 0:
           data = usb_serial.readline().decode('utf-8').strip()
           if data == "pass":
               pass_led.value = True
               time.sleep(5)
               pass_led.value = False
               usb_serial.write(b"Pass LED flashed\n")
           elif data == "fail":
               fail_led.value = True
               time.sleep(5)
               fail_led.value = False
               usb_serial.write(b"Fail LED flashed\n")
           else:
               usb_serial.write(b"Unknown command\n")
       time.sleep(0.1)

### 2. Set Up the Dockerized Application
1.	Clone or download this repository.
2.	Build the Docker image and run the container:

  ```sh
  cd reload
  docker build -t reload_app .
  docker run -p 7860:7860 --device=/dev/ttyACM1 reload_app
```

### 3. Access the Web Interface
Once the Docker container is running, access the Gradio web interface at:
```url
http://localhost:7860
```

### 4. Use the Application
1.	Upload an image through the Gradio interface.
2.	View the classification result (e.g., "Yes_Powder" or "No_Powder").
3.	Based on the result, the application will:
-	Send a "pass" command to the Pico if powder is detected, turning on the "Pass" LED.
-	Send a "fail" command otherwise, turning on the "Fail" LED.

## Wiring Diagram
<img src="https://imgur.com/HO8jAk3.jpg" alt="Wiring Diagram" width="300" height="600">


## Project Files
- app.py: Main application file that loads the model, processes images, and communicates with the Pico.
- Dockerfile: Docker configuration to build the application image.
- requirements.txt: Python dependencies for the project.

## Troubleshooting
- Pico Not Detected: If you see a message like Warning: /dev/ttyACM1 not found. Running without serial connection., ensure the Pico is connected and that Docker has permission to access it. Run the container with --device=/dev/ttyACM1.
- LEDs Not Lighting: Verify that the Pico code is running correctly and that the usb_cdc data channel is enabled.
- Model Loading Warning: FutureWarning messages may appear for PyTorch model loading. These warnings can generally be ignored.

## Future Enhancements
- Additional Classifications: Expand the model to classify more types of objects or materials.
- Notification System: Integrate an SMS or email alert system to notify users of the classification result.
- Web Interface Customization: Add more styling and functionalities to the Gradio web interface.

## License
This project is open source and available under the [MIT License](https://opensource.org/license/mit).

## Acknowledgments
- This project uses [Gradio](https://www.gradio.app/) for the web interface.
- Model developed with [PyTorch](https://pytorch.org/).
