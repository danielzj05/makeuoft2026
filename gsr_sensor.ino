/*
 * Grove GSR Sensor - Sweat/Stress Monitor for Lie Detection
 * ============================================================
 * Sensor: Grove GSR (SKU 101020052) by Seeed Studio
 * Connect: Grove connector to Arduino A0 (SIG pin)
 *          VCC → 5V (or 3.3V), GND → GND
 *
 * HOW THE SENSOR WORKS:
 * ─────────────────────
 * The GSR sensor measures electrical conductance across the skin.
 * Sweat contains electrolytes which conduct electricity.
 *
 *   HIGHER analog value  →  HIGHER skin conductance
 *                        →  MORE sweat being produced
 *                        →  STRONGER sympathetic nervous system activation
 *                        →  HIGHER stress / arousal / potential deception
 *
 *   LOWER analog value   →  LOWER skin conductance
 *                        →  LESS sweat / drier skin
 *                        →  CALMER / more relaxed state
 *
 * The sensor uses an op-amp (LM324) in a voltage divider configuration.
 * The two finger electrodes form part of that divider. As skin resistance
 * DECREASES (more sweat), output voltage — and therefore the analog read
 * value — INCREASES.
 *
 * NOTE: Raw values vary by individual. Always establish a baseline first.
 *
 * Wiring:
 *   Grove SIG → A0
 *   Grove VCC → 5V
 *   Grove GND → GND
 *   Finger straps → the two white connector leads on the sensor
 */

const int GSR_PIN       = A0;   // Analog input pin
const int SAMPLE_COUNT  = 10;   // Samples to average for noise reduction
const int BASELINE_SECS = 10;   // Seconds to collect baseline at startup

// Thresholds (as % rise above personal baseline)
// Adjust these after testing with your subject
const float THRESHOLD_MILD    = 0.05f;  // 5%  rise  → mild arousal
const float THRESHOLD_MODERATE = 0.15f; // 15% rise  → moderate stress
const float THRESHOLD_HIGH    = 0.30f;  // 30% rise  → high stress / deception flag

float baselineValue = 0.0f;
float peakValue     = 0.0f;
float minValue      = 1023.0f;

// ─── Helper: read smoothed GSR value ────────────────────────────────────────
float readGSR() {
  long sum = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sum += analogRead(GSR_PIN);
    delay(5);
  }
  return (float)sum / SAMPLE_COUNT;
}

// ─── Establish personal baseline ────────────────────────────────────────────
void calibrate() {
  Serial.println("=== GSR CALIBRATION ===");
  Serial.println("Sit still and breathe normally...");
  
  long   total    = 0;
  int    samples  = 0;
  unsigned long startTime = millis();
  
  while (millis() - startTime < (unsigned long)BASELINE_SECS * 1000) {
    float val = readGSR();
    total  += (long)val;
    samples++;
    Serial.print("Calibrating... ");
    Serial.println(val);
    delay(200);
  }
  
  baselineValue = (float)total / samples;
  peakValue     = baselineValue;
  minValue      = baselineValue;
  
  Serial.print("Baseline established: ");
  Serial.println(baselineValue);
  Serial.println("=======================");
  Serial.println("FORMAT: timestamp_ms, raw_value, delta_pct, stress_level");
}

// ─── Classify stress level ───────────────────────────────────────────────────
String classifyStress(float deltaPct) {
  if      (deltaPct >= THRESHOLD_HIGH)     return "HIGH";
  else if (deltaPct >= THRESHOLD_MODERATE) return "MODERATE";
  else if (deltaPct >= THRESHOLD_MILD)     return "MILD";
  else                                      return "CALM";
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  Serial.println("Grove GSR Sensor - Lie Detection Monitor");
  Serial.println("Attach finger straps to index and middle fingers.");
  delay(2000);
  calibrate();
}

// ─── Main loop ─────────────────────cl──────────────────────────────────────────
void loop() {
  float currentValue = readGSR();
  
  // Track peak and min since calibration
  if (currentValue > peakValue) peakValue = currentValue;
  if (currentValue < minValue)  minValue  = currentValue;
  
  // Delta as a fraction of baseline
  // Positive = MORE sweat than baseline (higher arousal)
  // Negative = LESS sweat than baseline (calmer than resting)
  float deltaPct = (currentValue - baselineValue) / baselineValue;
  
  String stressLevel = classifyStress(deltaPct);
  
  // Serial output (CSV — easy to graph in Serial Plotter or Python)
  //Serial.print(millis());
  //Serial.print(",");
  Serial.print(currentValue);
  Serial.print(",");
  Serial.print(deltaPct, 2);   // as percentage
  Serial.print(",");
  Serial.println(stressLevel);
  
  // Optional: visual alert on HIGH
  if (stressLevel == "HIGH") {
    // Uncomment if you have an LED on pin 13:
    // digitalWrite(LED_BUILTIN, HIGH);
    Serial.println(">>> DECEPTION FLAG: Significant stress response detected <<<");
  }
  
  delay(100);  // 10 Hz sample rate — fast enough to catch sudden spikes
}

/*
 * INTERPRETING VALUES FOR LIE DETECTION:
 * ──────────────────────────────────────
 * A sudden SPIKE in the raw value (and deltaPct) after a question is the
 * key indicator. You are looking for:
 *
 *  1. LATENCY: Spike usually occurs 1–4 seconds after a stressful stimulus.
 *     The Arduino timestamps help you correlate questions to responses.
 *
 *  2. MAGNITUDE: How much above baseline?
 *     - < 5%   → Normal variation, ignore
 *     - 5–15%  → Mild — may be reaction to any unexpected question
 *     - 15–30% → Moderate — noteworthy, context-dependent
 *     - > 30%  → Strong — significant sympathetic activation
 *
 *  3. RECOVERY TIME: Deceptive responses often show prolonged elevation.
 *     Innocent surprise spikes sharply and recovers quickly.
 *
 *  4. CONTROL QUESTIONS: Always ask known-true and known-false control
 *     questions first to calibrate THAT person's response range.
 *
 * IMPORTANT DISCLAIMER:
 * GSR alone is NOT a reliable lie detector. Stress ≠ deception.
 * Anxiety, surprise, embarrassment, or focus all raise GSR.
 * This is a research/educational tool only.
 */
