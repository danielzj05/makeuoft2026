import serial
import time
import sys

# CONFIGURATION
SERIAL_PORT = 'COM3'  # CHANGE THIS to your Arduino Port
BAUD_RATE = 115200
CALIBRATION_TIME = 10  # Seconds to measure baseline

def count_peaks(data_points, sampling_rate_est=0.02):
    """
    Simple algorithm to count heartbeats in a list of raw values.
    Returns the calculated BPM.
    """
    if not data_points:
        return 0
    
    # dynamic threshold: (Max + Min) / 2
    threshold = (max(data_points) + min(data_points)) / 2
    
    peaks = 0
    below_threshold = True
    
    for val in data_points:
        if below_threshold and val > threshold:
            peaks += 1
            below_threshold = False
        elif not below_threshold and val < threshold:
            below_threshold = True
            
    # Calculate BPM: (Peaks / Seconds) * 60
    duration_sec = len(data_points) * sampling_rate_est
    if duration_sec == 0: return 0
    
    bpm = (peaks / duration_sec) * 60
    return int(bpm)

# ---------------- MAIN PROGRAM ----------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial: {e}")
    sys.exit()

print("\n--- LOVE/LIE DETECTOR SETUP ---")
print("1. Relax and place finger on sensor.")
print("2. Wait a few seconds for the signal to stabilize.")
input("3. Press ENTER to start Calibration (10 seconds)...")

# --- CALIBRATION PHASE ---
print("\n[CALIBRATING] Do not move...")
start_time = time.time()
cal_data = []

ser.reset_input_buffer()

while time.time() - start_time < CALIBRATION_TIME:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.isdigit():
                cal_data.append(int(line))
        except:
            pass

# Analyze Calibration
if len(cal_data) == 0:
    print("Error: No data received from sensor!")
    sys.exit()

seconds_elapsed = time.time() - start_time
sample_rate = seconds_elapsed / len(cal_data) 
baseline_bpm = count_peaks(cal_data, sample_rate)

print(f"\n[DONE] Baseline Heart Rate: {baseline_bpm} BPM")
print("------------------------------------------------")
print("Starting Live Lie Detection... (Ctrl+C to stop)")
time.sleep(2)

# --- LIVE DETECTION LOOP ---
buffer = []
BUFFER_SIZE = 100  # Raw data window

# NEW: History for smoothing the BPM display
bpm_history = [] 
HISTORY_SIZE = 50  # How many recent BPM calculations to average

while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.isdigit():
                val = int(line)
                buffer.append(val)
                
                # Keep raw data buffer fixed size
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
                    
                    # 1. Calculate Instant BPM on this window
                    current_bpm = count_peaks(buffer, sample_rate)
                    
                    # 2. Add to history for smoothing
                    bpm_history.append(current_bpm)
                    if len(bpm_history) > HISTORY_SIZE:
                        bpm_history.pop(0)
                    
                    # 3. Calculate Sliding Window Average of BPM
                    if len(bpm_history) > 0:
                        avg_bpm = int(sum(bpm_history) / len(bpm_history))
                    else:
                        avg_bpm = 0
                    
                    # Logic: Check for Lie
                    status = "TRUTH"
                    # We compare the SMOOTHED average to baseline for better stability
                    # Or you can keep comparing current_bpm if you want it very sensitive
                    if avg_bpm > (baseline_bpm * 1.15):
                        status = "!!! LIE DETECTED !!!"
                    
                    # Display both Current (Instant) and Avg (Smoothed)
                    print(f"Base: {baseline_bpm} | Inst: {current_bpm} | Avg: {avg_bpm} | {status}", end='\r')
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            # Good to catch errors without crashing loop
            pass
