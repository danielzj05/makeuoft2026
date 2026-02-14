import serial
import time
import sys

# CONFIGURATION
SERIAL_PORT = 'COM5'   # Make sure this matches what worked for calibration!
BAUD_RATE = 115200
CALIBRATION_TIME = 10 

def count_peaks(data_points, sampling_rate_est=0.02):
    if not data_points: return 0
    
    threshold = (max(data_points) + min(data_points)) / 2
    peaks = 0
    below_threshold = True
    
    for val in data_points:
        if below_threshold and val > threshold:
            peaks += 1
            below_threshold = False
        elif not below_threshold and val < threshold:
            below_threshold = True
            
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
input("2. Press ENTER to start Calibration...")

# --- CALIBRATION PHASE ---
print("\n[CALIBRATING] Do not move...")
start_time = time.time()
cal_data = []

# Clear buffer once at start
ser.reset_input_buffer()

while time.time() - start_time < CALIBRATION_TIME:
    try:
        # forcing a read prevents "skipping" loops
        line = ser.readline().decode('utf-8').strip()
        if line.isdigit():
            cal_data.append(int(line))
    except:
        pass

if len(cal_data) == 0:
    print("Error: No data received during calibration.")
    sys.exit()

# Calculate Baseline
seconds_elapsed = time.time() - start_time
sample_rate = seconds_elapsed / len(cal_data) 
baseline_bpm = count_peaks(cal_data, sample_rate)

print(f"\n[DONE] Baseline Heart Rate: {baseline_bpm} BPM")
print("------------------------------------------------")
print("Starting Live Lie Detection...")
print("(Updates once per second to prevent crashing)")
time.sleep(1)

# --- LIVE DETECTION LOOP ---
buffer = []
BUFFER_SIZE = 100 
bpm_history = []
HISTORY_SIZE = 50 

counter = 0 # Counter to slow down printing

while True:
    try:
        # Blocking read - waits for data
        line = ser.readline().decode('utf-8').strip()
        
        if line.isdigit():
            val = int(line)
            buffer.append(val)
            
            # 1. Fill the Buffer
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)
                
                # 2. Process Data
                current_bpm = count_peaks(buffer, sample_rate)
                
                bpm_history.append(current_bpm)
                if len(bpm_history) > HISTORY_SIZE:
                    bpm_history.pop(0)
                
                avg_bpm = int(sum(bpm_history) / len(bpm_history))
                
                # 3. Lie Logic
                status = "TRUTH"
                if avg_bpm > (baseline_bpm * 1.15):
                    status = "!!! LIE DETECTED !!!"
                
                # 4. PRINT ONLY ONCE EVERY 50 SAMPLES (Approx 1 second)
                counter += 1
                if counter >= 50:
                    print(f"Base: {baseline_bpm} | Inst: {current_bpm} | Avg: {avg_bpm} | {status}")
                    counter = 0
            
            else:
                # While buffer is filling, print every 10 samples so you see progress
                if len(buffer) % 10 == 0:
                    print(f"Buffering... {len(buffer)}/{BUFFER_SIZE}")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        # Print error but keep going
        print(f"Error: {e}")
