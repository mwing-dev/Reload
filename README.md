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
   
3. Copy the main code to code.py on the Pico:
   
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

jhjhj
