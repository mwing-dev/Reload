<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification - Pass/Fail with Raspberry Pi Pico</title>
</head>
<body>
    <h1>Image Classification - Pass/Fail with Raspberry Pi Pico</h1>

    <p>This project uses an image classification model to detect the presence of a specific substance (e.g., powder) in an image, returning a "Pass" or "Fail" result. The classification result is sent to a Raspberry Pi Pico microcontroller, which lights up specific LEDs to indicate the result.</p>

    <h2>Features</h2>
    <ul>
        <li><strong>Image Classification</strong>: A CNN model classifies images into "Yes_Powder" or "No_Powder".</li>
        <li><strong>Real-Time Feedback</strong>: Based on the classification result, the system sends commands to a Raspberry Pi Pico via USB.</li>
        <li><strong>LED Indicators</strong>: The Pico activates a "Pass" LED if powder is detected or a "Fail" LED otherwise.</li>
    </ul>

    <h2>Project Components</h2>
    <ul>
        <li><strong>Dockerized Python Application</strong>: Runs the Gradio-based web interface and communicates with the Pico.</li>
        <li><strong>Raspberry Pi Pico</strong>: Controls LEDs based on received commands.</li>
        <li><strong>Gradio Interface</strong>: Web interface to upload images and view classification results.</li>
    </ul>

    <h2>Requirements</h2>
    <h3>Hardware</h3>
    <ul>
        <li>Raspberry Pi Pico with CircuitPython installed.</li>
        <li>Two LEDs connected to GPIO pins 14 (Fail) and 15 (Pass) on the Pico.</li>
        <li>USB connection between the Pico and the host machine.</li>
    </ul>

    <h3>Software</h3>
    <ul>
        <li>Docker and Docker Compose installed on the host machine.</li>
        <li>Python packages (see <code>requirements.txt</code>): <code>torch</code>, <code>torchvision</code>, <code>Pillow</code>, <code>gradio</code>, <code>pyserial</code>.</li>
        <li>Model weights in a file named <code>Trained_Model.pth</code> located in <code>/app/Reload_Trained_Model</code>.</li>
    </ul>

    <h2>Setup and Installation</h2>

    <h3>1. Set Up Raspberry Pi Pico</h3>
    <ol>
        <li>Install <strong>CircuitPython</strong> on the Pico.</li>
        <li>Copy the following code to <code>boot.py</code> on the Pico to enable USB data communication:</li>
    </ol>
    <pre><code>import usb_cdc
usb_cdc.enable(data=True)
</code></pre>

    <ol start="3">
        <li>Copy the main code to <code>code.py</code> on the Pico:</li>
    </ol>
    <pre><code>import time
import digitalio
import board
import usb_cdc

pass_led = digitalio.DigitalInOut(board.GP15)
pass_led.direction = digitalio.Direction.OUTPUT
fail_led = digitalio.DigitalInOut(board.GP14)
fail_led.direction = digitalio.Direction.OUTPUT

usb_serial = usb_cdc.data

while True:
    if usb_serial and usb_serial.in_waiting &gt; 0:
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
</code></pre>

    <h3>2. Set Up the Dockerized Application</h3>
    <ol>
        <li>Clone or download this repository.</li>
        <li>Place <code>Trained_Model.pth</code> in <code>/app/Reload_Trained_Model</code>.</li>
        <li>Build the Docker image and run the container:</li>
    </ol>
    <pre><code>cd reload
docker build -t reload_app .
docker run -p 7860:7860 --device=/dev/ttyACM1 reload_app
</code></pre>

    <h3>3. Access the Web Interface</h3>
    <p>Once the Docker container is running, access the Gradio web interface at:</p>
    <code>http://localhost:7860</code>

    <h3>4. Use the Application</h3>
    <ol>
        <li>Upload an image through the Gradio interface.</li>
        <li>View the classification result (e.g., "Yes_Powder" or "No_Powder").</li>
        <li>Based on the result, the application will:
            <ul>
                <li>Send a <code>"pass"</code> command to the Pico if powder is detected, turning on the "Pass" LED.</li>
                <li>Send a <code>"fail"</code> command otherwise, turning on the "Fail" LED.</li>
            </ul>
        </li>
    </ol>

    <h2>Project Files</h2>
    <ul>
        <li><strong>app.py</strong>: Main application file that loads the model, processes images, and communicates with the Pico.</li>
        <li><strong>Dockerfile</strong>: Docker configuration to build the application image.</li>
        <li><strong>requirements.txt</strong>: Python dependencies for the project.</li>
        <li><strong>Trained_Model.pth</strong>: Pre-trained weights for the image classification model (not included in this repository).</li>
    </ul>

    <h2>Troubleshooting</h2>
    <ul>
        <li><strong>Pico Not Detected</strong>: If you see a message like <code>Warning: /dev/ttyACM1 not found. Running without serial connection.</code>, ensure the Pico is connected and that Docker has permission to access it. Run the container with <code>--device=/dev/ttyACM1</code>.</li>
        <li><strong>LEDs Not Lighting</strong>: Verify that the Pico code is running correctly and that the <code>usb_cdc</code> data channel is enabled.</li>
        <li><strong>Model Loading Warning</strong>: <code>FutureWarning</code> messages may appear for PyTorch model loading. These warnings can generally be ignored.</li>
    </ul>

    <h2>Future Enhancements</h2>
    <ul>
        <li><strong>Additional Classifications</strong>: Expand the model to classify more types of objects or materials.</li>
        <li><strong>Notification System</strong>: Integrate an SMS or email alert system to notify users of the classification result.</li>
        <li><strong>Web Interface Customization</strong>: Add more styling and functionalities to the Gradio web interface.</li>
    </ul>

    <h2>License</h2>
    <p>This project is open source and available under the <a href="LICENSE">MIT License</a>.</p>

    <h2>Acknowledgments</h2>
    <ul>
        <li>This project uses <a href="https://gradio.app/">Gradio</a> for the web interface.</li>
        <li>Model developed with <a href="https://pytorch.org/">PyTorch</a>.</li>
    </ul>
</body>
</html>
