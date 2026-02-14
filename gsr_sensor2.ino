/*
 * Grove GSR Sensor - Lie Detection Monitor
 * =========================================
 * Sensor: Grove GSR (SKU 101020052) — SIG pin → A0
 *
 * ── YOUR SENSOR'S BEHAVIOUR ──────────────────────────────────────────────────
 * Resting baseline:  ~200–400 ADC
 * Stress response:   value DROPS (potentiometer tuned high, inverts signal)
 *
 *   LOWER value  →  more sweat  →  more stress / arousal / potential deception
 *   HIGHER value →  calmer / drier skin
 *
 * ── DETECTION STRATEGY ───────────────────────────────────────────────────────
 * Problem with a static baseline: skin drifts slowly over a session, so after
 * a few minutes the baseline is stale and deltas become meaningless.
 *
 * Solution: ROLLING BASELINE using exponential moving average (EMA).
 * The baseline slowly tracks natural drift (time constant ~30 s) but cannot
 * follow a sudden stress spike — so genuine reactions stand out clearly.
 *
 * Two independent detectors, both must trigger for a HIGH flag:
 *
 *   1. SPIKE detector  — did the value drop ≥ SPIKE_THRESH points within the
 *                        last SPIKE_WINDOW samples? Catches the initial burst.
 *
 *   2. SUSTAINED detector — has the value stayed ≥ SUSTAIN_THRESH below the
 *                           rolling baseline for ≥ SUSTAIN_SAMPLES in a row?
 *                           Separates a real response from a cough/fidget.
 *
 * Requiring BOTH filters out false positives from physical movement.
 *
 * ── WIRING ───────────────────────────────────────────────────────────────────
 *   Grove SIG  → A0
 *   Grove VCC  → 5V (or 3.3V)
 *   Grove GND  → GND
 *   Finger straps on index + middle finger, same hand
 */

// ── Pin & sampling ────────────────────────────────────────────────────────────
const int   GSR_PIN       = A0;
const int   SAMPLE_COUNT  = 8;     // readings averaged per tick (noise filter)
const int   LOOP_MS       = 100;   // tick rate = 10 Hz

// ── Rolling baseline (EMA) ────────────────────────────────────────────────────
// Alpha controls how fast baseline follows drift.
// 0.005 ≈ 30-second time constant at 10 Hz — slow enough to ignore stress spikes,
// fast enough to handle natural skin drift over a session.
const float EMA_ALPHA     = 0.005f;

// ── Spike detector ────────────────────────────────────────────────────────────
// Look back SPIKE_WINDOW ticks (2 s) for a drop of ≥ SPIKE_THRESH raw units.
// Hyperventilating drops ~100 pts; a lie response is typically 15–40 pts.
const int   SPIKE_WINDOW  = 20;    // ticks to look back  (20 × 100ms = 2 s)
const float SPIKE_THRESH  = 18.0f; // minimum drop in raw ADC to call a spike
                                   // ↑ raise if too many false positives
                                   // ↓ lower if real responses are being missed

// ── Sustained detector ────────────────────────────────────────────────────────
// Must stay ≥ SUSTAIN_THRESH below rolling baseline for ≥ SUSTAIN_SAMPLES ticks.
const float SUSTAIN_THRESH   = 15.0f; // raw ADC below baseline
const int   SUSTAIN_SAMPLES  = 15;    // must hold for 1.5 s (15 × 100ms)

// ── State ─────────────────────────────────────────────────────────────────────
float rollingBaseline  = -1.0f;   // initialised on first reading
float recentBuf[20];              // circular buffer for spike window
int   bufIndex         = 0;
bool  bufFull          = false;
int   sustainCount     = 0;       // consecutive ticks below threshold
bool  spikeActive      = false;
bool  sustainActive    = false;
int   flagCooldown     = 0;       // ticks before next HIGH flag can fire
const int FLAG_COOLDOWN_TICKS = 30; // 3 s cooldown between flags

// ── Read smoothed value ───────────────────────────────────────────────────────
float readGSR() {
  long sum = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sum += analogRead(GSR_PIN);
    delayMicroseconds(500);
  }
  return (float)sum / SAMPLE_COUNT;
}

// ── Calibrate: seed the rolling baseline ─────────────────────────────────────
void calibrate() {
  Serial.println("=== CALIBRATION: sit still, breathe normally ===");
  long total = 0;
  for (int i = 0; i < 50; i++) {          // 5 seconds at 10 Hz
    total += (long)readGSR();
    delay(LOOP_MS);
  }
  rollingBaseline = (float)total / 50.0f;
  // Pre-fill spike buffer with baseline so first ticks don't false-fire
  for (int i = 0; i < SPIKE_WINDOW; i++) recentBuf[i] = rollingBaseline;
  bufIndex = 0; bufFull = true;

  Serial.print("Baseline: ");
  Serial.println(rollingBaseline);
  Serial.println("=================================================");
  Serial.println("ts_ms,raw,rolling_baseline,drop_from_baseline,spike,sustained,level");
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("Grove GSR - Lie Detection Monitor");
  delay(1000);
  calibrate();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  float raw = readGSR();

  // ── 1. Update rolling baseline (EMA) ──────────────────────────────────────
  // Only update if value is ABOVE (calmer than) or very close to baseline.
  // This prevents a stress dip from dragging the baseline down with it.
  if (raw >= rollingBaseline - SUSTAIN_THRESH) {
    rollingBaseline = (EMA_ALPHA * raw) + ((1.0f - EMA_ALPHA) * rollingBaseline);
  }

  // ── 2. Spike detector ─────────────────────────────────────────────────────
  // Store in circular buffer
  recentBuf[bufIndex] = raw;
  bufIndex = (bufIndex + 1) % SPIKE_WINDOW;
  if (bufIndex == 0) bufFull = true;

  // Find max value in the window (the "recent calm" reference)
  float windowMax = raw;
  int len = bufFull ? SPIKE_WINDOW : bufIndex;
  for (int i = 0; i < len; i++) {
    if (recentBuf[i] > windowMax) windowMax = recentBuf[i];
  }
  float spikeDrop = windowMax - raw;
  spikeActive = (spikeDrop >= SPIKE_THRESH);

  // ── 3. Sustained detector ─────────────────────────────────────────────────
  float dropFromBaseline = rollingBaseline - raw;  // positive = below baseline
  if (dropFromBaseline >= SUSTAIN_THRESH) {
    sustainCount++;
  } else {
    sustainCount = 0;
  }
  sustainActive = (sustainCount >= SUSTAIN_SAMPLES);

  // ── 4. Classify ───────────────────────────────────────────────────────────
  String level = "CALM";
  if      (spikeActive && sustainActive && flagCooldown == 0) level = "HIGH";
  else if (sustainActive)                                      level = "MODERATE";
  else if (spikeActive)                                        level = "MILD";

  // ── 5. LED + cooldown ─────────────────────────────────────────────────────
  if (level == "HIGH") {
    digitalWrite(LED_BUILTIN, HIGH);
    flagCooldown = FLAG_COOLDOWN_TICKS;
    Serial.println(">>> DECEPTION FLAG <<<");
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
  if (flagCooldown > 0) flagCooldown--;

  // ── 6. Serial output (CSV for Serial Plotter or Python) ───────────────────
 // Serial.print(millis());      Serial.print(",");
  Serial.print(raw);           Serial.print(",");
  //Serial.print(rollingBaseline); Serial.print(",");
  //Serial.print(dropFromBaseline); Serial.print(",");
  //Serial.print(spikeActive ? 1 : 0); Serial.print(",");
  //Serial.print(sustainActive ? 1 : 0); Serial.print(",");
  Serial.println(level);

  delay(LOOP_MS);
}

/*
 * ── TUNING GUIDE ─────────────────────────────────────────────────────────────
 *
 * Getting too many false positives (flags when calm)?
 *   → Raise SPIKE_THRESH  (try 25–35)
 *   → Raise SUSTAIN_THRESH (try 20–25)
 *   → Raise SUSTAIN_SAMPLES (try 20 → 2 seconds)
 *
 * Missing real responses (no flag when lying)?
 *   → Lower SPIKE_THRESH  (try 12–15)
 *   → Lower SUSTAIN_THRESH (try 10)
 *   → Lower SUSTAIN_SAMPLES (try 10 → 1 second)
 *
 * Baseline drifting too fast (adapting to stress)?
 *   → Lower EMA_ALPHA (try 0.002)
 *
 * Baseline drifting too slow (stale after moving sensor)?
 *   → Raise EMA_ALPHA (try 0.01)
 *
 * ── CONTROL QUESTION PROTOCOL ────────────────────────────────────────────────
 * Always run this before testing for lies:
 *
 *  1. Ask 3 questions the subject MUST answer truthfully (name, birthday, etc.)
 *     Watch the drop magnitude — this is their "truth baseline response"
 *
 *  2. Ask 1 question they MUST lie about (pre-arranged)
 *     Note the drop — this is their personal lie signature
 *
 *  3. Now calibrate SPIKE_THRESH to sit between those two values.
 *     Every person is different. Don't skip this step.
 *
 * ── READING THE CSV ──────────────────────────────────────────────────────────
 *  ts_ms              — milliseconds since startup
 *  raw                — smoothed ADC reading (lower = more stress for your sensor)
 *  rolling_baseline   — slow-moving personal normal
 *  drop_from_baseline — how far below baseline right now (positive = stressed)
 *  spike              — 1 if sudden drop detected in last 2 s
 *  sustained          — 1 if held below threshold for 1.5 s
 *  level              — CALM / MILD / MODERATE / HIGH
 *
 * NOTE: Not a medical device. GSR measures arousal, not deception specifically.
 */
