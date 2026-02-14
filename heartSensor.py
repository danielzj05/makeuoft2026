# --- LIVE DETECTION LOOP ---
buffer = []
BUFFER_SIZE = 100 # Window size for raw data
bpm_history = []
HISTORY_SIZE = 50 

print(f"Base: {baseline_bpm} BPM | Waiting for buffer to fill...")

while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.isdigit():
                val = int(line)
                buffer.append(val)
                
                # Keep buffer size fixed
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
                    
                    # --- CALCULATE & DISPLAY ---
                    
                    # 1. Instant BPM
                    current_bpm = count_peaks(buffer, sample_rate)
                    
                    # 2. Smooth Average BPM
                    bpm_history.append(current_bpm)
                    if len(bpm_history) > HISTORY_SIZE:
                        bpm_history.pop(0)
                        
                    avg_bpm = int(sum(bpm_history) / len(bpm_history))
                    
                    # 3. Lie Logic
                    status = "TRUTH"
                    if avg_bpm > (baseline_bpm * 1.15):
                        status = "!!! LIE DETECTED !!!"
                    
                    # PRINT with flush=True to ensure it updates instantly
                    print(f"Base: {baseline_bpm} | Inst: {current_bpm} | Avg: {avg_bpm} | {status}   ", end='\r', flush=True)
                
                else:
                    # Optional: Print progress while filling the buffer so you know it's working
                    pct = int((len(buffer) / BUFFER_SIZE) * 100)
                    print(f"Reading sensor... {pct}% buffer filled", end='\r', flush=True)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            # If something crashes, print the error so we know why!
            print(f"\nError: {e}")
