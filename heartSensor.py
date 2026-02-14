import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- CONFIGURATION ---
# CHANGE 'COM3' to your actual port (e.g., '/dev/ttyUSB0' on Mac)
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200 # Must match your Arduino Serial.begin()

# --- SETUP ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except:
    print("Error: Could not open serial port. Is the Arduino IDE Serial Monitor/Plotter open? Close it!")
    exit()

# Data storage
data_heart = deque([0] * 200, maxlen=200)

# Setup Plot
fig, ax = plt.subplots()
line, = ax.plot(data_heart, color='red')
ax.set_ylim(0, 1024)
ax.set_title("Live Heart Rate Signal")

def animate(i):
    if ser.in_waiting:
        try:
            # Read line: "Signal 512"
            line = ser.readline().decode('utf-8').strip()
            
            # Parse logic: Split by space and take the last part
            # "Signal 512" -> ["Signal", "512"] -> 512
            if "Signal" in line:
                val_str = line.split(" ")[-1] # Grab the number part
                val = int(val_str)
                
                data_heart.append(val)
                line.set_ydata(data_heart)
        except ValueError:
            pass
    return line,

ani = animation.FuncAnimation(fig, animate, interval=20)
plt.show()