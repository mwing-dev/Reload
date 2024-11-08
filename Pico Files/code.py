import time
import digitalio
import board
import usb_cdc

# Set up GPIO pins for LEDs
pass_led = digitalio.DigitalInOut(board.GP15)
pass_led.direction = digitalio.Direction.OUTPUT
fail_led = digitalio.DigitalInOut(board.GP14)
fail_led.direction = digitalio.Direction.OUTPUT

# Listening for data from USB
usb_serial = usb_cdc.data  # Use the data channel

print("Listening for data...")

while True:
    if usb_serial and usb_serial.in_waiting > 0:  # Check that usb_serial is not None
        try:
            # Read incoming data
            data = usb_serial.readline().decode('utf-8').strip()
            print("Received:", data)

            # Process the command
            if data == "pass":
                print("Triggering Pass LED")
                pass_led.value = True
                time.sleep(5)
                pass_led.value = False
                usb_serial.write(b"Pass LED flashed\n")
            elif data == "fail":
                print("Triggering Fail LED")
                fail_led.value = True
                time.sleep(5)
                fail_led.value = False
                usb_serial.write(b"Fail LED flashed\n")
            else:
                print("Received unknown command")
                usb_serial.write(b"Unknown command\n")
        
        except Exception as e:
            print(f"Error reading from USB: {e}")
            usb_serial.write(b"Error reading command\n")

    time.sleep(0.1)  # Small delay to prevent excessive CPU usage