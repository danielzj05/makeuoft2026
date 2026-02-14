import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from collections import deque

# ========================== CONFIG ==========================
MODEL_PATH = 'face_landmarker.task'
BASELINE_FRAMES = 150          # ~5s @ 30fps — frames to collect during LEARNING
ROLLING_WINDOW = 30            # rolling avg window for smoothing live signal
GRAPH_LEN = 300                # how many samples the bottom graph shows
SPIKE_WINDOW = 8               # frames to detect temporal micro-expression spikes
BLINK_RATE_WINDOW = 150        # frames over which to compute blink rate (~5s)
GAZE_JITTER_WINDOW = 20        # frames for gaze stability measurement

# Thresholds
COMBINED_THRESHOLD = 0.45          # final weighted verdict score
SMILE_SUPPRESS_THRESHOLD = 0.4     # smile level above which micro-expr is dampened

# Verdict channel weights (sum to 1.0)
W_IRIS       = 0.20    # iris dilation — useful but noisy
W_MICRO      = 0.15    # raw micro-expression (dampened by smile filter)
W_ASYMMETRY  = 0.25    # facial asymmetry — very hard to fake
W_BLINK      = 0.20    # blink rate deviation — involuntary
W_GAZE       = 0.10    # gaze instability
W_SPIKE      = 0.10    # temporal micro-expression spikes (brief flashes)

# Micro-expression weights (stress/deception-related blendshapes)
MICRO_WEIGHTS = {
    'browDownLeft':      0.20,
    'browDownRight':     0.20,
    'browInnerUp':       0.15,   # surprise / cognitive load
    'eyeSquintLeft':     0.15,
    'eyeSquintRight':    0.15,
    'noseSneerLeft':     0.18,
    'noseSneerRight':    0.18,
    'mouthPressLeft':    0.20,   # lip press — strong lie indicator
    'mouthPressRight':   0.20,
    'mouthDimpleLeft':   0.10,
    'mouthDimpleRight':  0.10,
    'jawClench':         0.12,
}

# Paired blendshapes for asymmetry detection (left, right)
ASYMMETRY_PAIRS = [
    ('browDownLeft',     'browDownRight'),
    ('eyeSquintLeft',    'eyeSquintRight'),
    ('noseSneerLeft',    'noseSneerRight'),
    ('mouthSmileLeft',   'mouthSmileRight'),
    ('mouthFrownLeft',   'mouthFrownRight'),
    ('mouthPressLeft',   'mouthPressRight'),
    ('mouthDimpleLeft',  'mouthDimpleRight'),
    ('cheekSquintLeft',  'cheekSquintRight'),
    ('mouthUpperUpLeft', 'mouthUpperUpRight'),
    ('mouthLowerDownLeft','mouthLowerDownRight'),
    ('browOuterUpLeft',  'browOuterUpRight'),
]

# Smile indicators (used to suppress false positives from laughing)
SMILE_SHAPES = ['mouthSmileLeft', 'mouthSmileRight', 'cheekPuff']

# ======================== LANDMARK INDICES ========================
LEFT_EYE_TOP = 159
LEFT_EYE_BOT = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOT = 374
# Iris centers
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
# Eye corners for gaze normalization
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


# ======================== HELPER FUNCTIONS ========================

def lm_to_px(lm, w, h):
    """Convert a normalized landmark to pixel coords."""
    return np.array([lm.x * w, lm.y * h])


def iris_area(landmarks, w, h, side='left'):
    """
    Compute iris 'diamond' area from 4 cardinal iris landmarks.
    Purely geometric — works for dark eyes.
    """
    if side == 'left':
        r, t, l, b = 469, 470, 471, 472
    else:
        r, t, l, b = 474, 475, 476, 477
    pts = np.array([lm_to_px(landmarks[i], w, h) for i in [r, t, l, b]])
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def eye_opening(landmarks, w, h, side='left'):
    """Vertical distance between upper and lower eyelid (in pixels)."""
    if side == 'left':
        top_idx, bot_idx = LEFT_EYE_TOP, LEFT_EYE_BOT
    else:
        top_idx, bot_idx = RIGHT_EYE_TOP, RIGHT_EYE_BOT
    return np.linalg.norm(lm_to_px(landmarks[top_idx], w, h) -
                          lm_to_px(landmarks[bot_idx], w, h))


def normalized_iris_metric(landmarks, w, h):
    """
    Iris area / eye_opening^2.  Cancels distance-to-camera.
    Independent of iris/pupil color.
    """
    l_area = iris_area(landmarks, w, h, 'left')
    r_area = iris_area(landmarks, w, h, 'right')
    l_open = max(eye_opening(landmarks, w, h, 'left'), 1e-6)
    r_open = max(eye_opening(landmarks, w, h, 'right'), 1e-6)
    l_ratio = l_area / (l_open ** 2)
    r_ratio = r_area / (r_open ** 2)
    return (l_ratio + r_ratio) / 2.0


def gaze_position(landmarks, w, h):
    """
    Return normalized gaze position for each eye: how far the iris center
    is from the eye center, as a fraction of the eye width.
    Returns average of both eyes.
    """
    positions = []
    for side in ['left', 'right']:
        if side == 'left':
            iris_c = LEFT_IRIS_CENTER
            inner, outer = LEFT_EYE_INNER, LEFT_EYE_OUTER
        else:
            iris_c = RIGHT_IRIS_CENTER
            inner, outer = RIGHT_EYE_INNER, RIGHT_EYE_OUTER

        ic = lm_to_px(landmarks[iris_c], w, h)
        ei = lm_to_px(landmarks[inner], w, h)
        eo = lm_to_px(landmarks[outer], w, h)
        eye_w = max(np.linalg.norm(ei - eo), 1e-6)
        eye_center = (ei + eo) / 2.0
        offset = np.linalg.norm(ic - eye_center) / eye_w
        positions.append(offset)
    return np.mean(positions)


def brightness_of_frame(frame):
    """Mean brightness (0-255) of the frame in grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def micro_expression_score(blendshapes):
    """
    Weighted sum of stress/deception-related blendshapes.
    Returns a 0-1 score; higher = more suspicious.
    """
    score = 0.0
    total_weight = 0.0
    for name, weight in MICRO_WEIGHTS.items():
        score += blendshapes.get(name, 0.0) * weight
        total_weight += weight
    return score / total_weight if total_weight > 0 else 0.0


def smile_level(blendshapes):
    """How much the person is smiling/laughing (0-1)."""
    return max(blendshapes.get(s, 0.0) for s in SMILE_SHAPES)


def asymmetry_score(blendshapes):
    """
    Average absolute difference between paired left/right blendshapes.
    Genuine emotions are bilaterally symmetric. Suppressed or fake
    emotions create asymmetry. This is nearly impossible to control
    consciously.
    """
    diffs = []
    for left_name, right_name in ASYMMETRY_PAIRS:
        lv = blendshapes.get(left_name, 0.0)
        rv = blendshapes.get(right_name, 0.0)
        # Only count if at least one side is active (not resting face)
        if lv > 0.05 or rv > 0.05:
            diffs.append(abs(lv - rv))
    return np.mean(diffs) if diffs else 0.0


def detect_spike(buffer, window=SPIKE_WINDOW):
    """
    Detect if there was a brief spike in the recent buffer.
    A 'spike' = the max in the last `window` frames is significantly
    higher than the frames just before it. Catches suppressed
    micro-expressions that flash for a fraction of a second.
    """
    if len(buffer) < window * 2:
        return 0.0
    recent = list(buffer)[-window:]
    prior = list(buffer)[-(window * 2):-window]
    recent_max = max(recent)
    prior_mean = np.mean(prior)
    prior_std = max(np.std(prior), 1e-6)
    spike_z = (recent_max - prior_mean) / prior_std
    # The spike must also have subsided (brief, not sustained)
    recent_min = min(recent[-3:]) if len(recent) >= 3 else recent[-1]
    is_brief = (recent_max - recent_min) > 0.05
    return min(max(spike_z, 0) / 4.0, 1.0) if is_brief else 0.0


def zscore_to_01(value, mean, std, cap_stds=3.0):
    """Convert a value to a 0-1 score based on z-score deviation."""
    z = abs(value - mean) / max(std, 1e-6)
    return min(z / cap_stds, 1.0)


def draw_gauge(frame, x, y, w_bar, h_bar, value, label, color):
    """Draw a horizontal bar gauge on the frame."""
    cv2.rectangle(frame, (x, y), (x + w_bar, y + h_bar), (60, 60, 60), -1)
    fill = int(np.clip(value, 0, 1) * w_bar)
    cv2.rectangle(frame, (x, y), (x + fill, y + h_bar), color, -1)
    cv2.rectangle(frame, (x, y), (x + w_bar, y + h_bar), (180, 180, 180), 1)
    cv2.putText(frame, f'{label}: {value:.2f}', (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


# ======================== INIT ========================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ======================== STATE ========================
mode = "LEARNING"              # LEARNING | INTERROGATION

# Baseline collectors
baseline_iris       = deque(maxlen=BASELINE_FRAMES)
baseline_brightness = deque(maxlen=BASELINE_FRAMES)
baseline_micro      = deque(maxlen=BASELINE_FRAMES)
baseline_asymmetry  = deque(maxlen=BASELINE_FRAMES)
baseline_gaze       = deque(maxlen=BASELINE_FRAMES)
baseline_blink_ts   = deque(maxlen=500)  # timestamps of blinks during learning

# Locked baseline stats
locked = {}  # will hold mean/std for each channel

# Live rolling buffers
rolling_iris      = deque(maxlen=ROLLING_WINDOW)
rolling_micro     = deque(maxlen=ROLLING_WINDOW)
rolling_asymmetry = deque(maxlen=ROLLING_WINDOW)
rolling_gaze      = deque(maxlen=GAZE_JITTER_WINDOW)

# Blink rate tracking
blink_timestamps  = deque(maxlen=500)
was_blinking      = False

# Micro-expression spike detection buffer
micro_raw_buf = deque(maxlen=SPIKE_WINDOW * 3)

# Graph history
iris_history      = deque(maxlen=GRAPH_LEN)
micro_history     = deque(maxlen=GRAPH_LEN)
asymmetry_history = deque(maxlen=GRAPH_LEN)
blink_history     = deque(maxlen=GRAPH_LEN)
gaze_history      = deque(maxlen=GRAPH_LEN)
verdict_history   = deque(maxlen=GRAPH_LEN)

frame_counter = 0

print("=" * 55)
print("  MakeUofT 2026 — AI Polygraph (Valentine's Edition)")
print("=" * 55)
print("DETECTION CHANNELS:")
print("  1. Iris geometry   (pupil dilation, dark-eye safe)")
print("  2. Micro-expression (smile-filtered, won't false-positive on laughing)")
print("  3. Facial asymmetry (genuine vs fake emotion — can't be faked)")
print("  4. Blink rate      (involuntary stress response)")
print("  5. Gaze stability  (eye jitter under cognitive load)")
print("  6. Temporal spikes (brief suppressed micro-expressions)")
print("-" * 55)
print("CONTROLS:")
print("  SPACE  — Lock baseline & start interrogation")
print("  R      — Reset to learning mode")
print("  Q      — Quit")
print("=" * 55)
print("Ask 3 TRUTH questions first to calibrate.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    ih, iw, _ = frame.shape
    frame_counter += 1
    now = time.time()

    timestamp_ms = int(frame_counter * (1000 / 30))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    status_text = "NO FACE DETECTED"
    status_color = (120, 120, 120)
    verdict_val = 0.0

    if result.face_landmarks and result.face_blendshapes:
        landmarks = result.face_landmarks[0]
        blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
        blink_l = blendshapes.get('eyeBlinkLeft', 0)
        blink_r = blendshapes.get('eyeBlinkRight', 0)
        is_blinking = blink_l > 0.5 or blink_r > 0.5

        # Track blink events (rising edge)
        if is_blinking and not was_blinking:
            blink_timestamps.append(now)
            if mode == "LEARNING":
                baseline_blink_ts.append(now)
        was_blinking = is_blinking

        frame_bright = brightness_of_frame(frame)

        if not is_blinking:
            # ---- Compute all channels ----
            raw_iris = normalized_iris_metric(landmarks, iw, ih)
            raw_micro = micro_expression_score(blendshapes)
            raw_asym = asymmetry_score(blendshapes)
            raw_gaze = gaze_position(landmarks, iw, ih)
            raw_smile = smile_level(blendshapes)

            # Smile filter: dampen micro-expression when genuinely smiling
            if raw_smile > SMILE_SUPPRESS_THRESHOLD:
                raw_micro *= max(0.0, 1.0 - raw_smile)  # suppress toward 0

            rolling_iris.append(raw_iris)
            rolling_micro.append(raw_micro)
            rolling_asymmetry.append(raw_asym)
            rolling_gaze.append(raw_gaze)
            micro_raw_buf.append(raw_micro)

            smooth_iris = np.mean(rolling_iris)
            smooth_micro = np.mean(rolling_micro)
            smooth_asym = np.mean(rolling_asymmetry)

            # Gaze jitter = std of recent gaze positions (higher = less stable)
            gaze_jitter = np.std(rolling_gaze) if len(rolling_gaze) >= 5 else 0.0

            # Blink rate (blinks per minute over recent window)
            recent_blinks = [t for t in blink_timestamps if now - t < 5.0]
            blink_rate = len(recent_blinks) * 12.0  # scale 5s window to per-minute

            # ---- LEARNING MODE ----
            if mode == "LEARNING":
                baseline_iris.append(raw_iris)
                baseline_brightness.append(frame_bright)
                baseline_micro.append(raw_micro)
                baseline_asymmetry.append(raw_asym)
                baseline_gaze.append(gaze_jitter)

                pct = len(baseline_iris) * 100 // BASELINE_FRAMES
                status_text = f"LEARNING BASELINE... {pct}%"
                status_color = (255, 200, 0)

                if pct >= 100:
                    status_text = "BASELINE READY - press SPACE"
                    status_color = (0, 255, 200)

            # ---- INTERROGATION MODE ----
            elif mode == "INTERROGATION" and locked:
                # Brightness compensation for iris
                bright_ratio = frame_bright / max(locked['brightness'], 1e-6)
                comp_iris = smooth_iris / max(bright_ratio, 0.5)

                # Score each channel (0-1)
                s_iris  = zscore_to_01(comp_iris, locked['iris_m'], locked['iris_s'])
                s_micro = zscore_to_01(smooth_micro, locked['micro_m'], locked['micro_s'])
                s_asym  = zscore_to_01(smooth_asym, locked['asym_m'], locked['asym_s'])
                s_gaze  = zscore_to_01(gaze_jitter, locked['gaze_m'], locked['gaze_s'])
                s_spike = detect_spike(micro_raw_buf)

                # Blink rate score
                blink_dev = abs(blink_rate - locked['blink_rate']) / max(locked['blink_rate'], 1e-6)
                s_blink = min(blink_dev / 1.0, 1.0)  # 100% deviation = 1.0

                # Weighted verdict
                verdict_val = (W_IRIS * s_iris +
                               W_MICRO * s_micro +
                               W_ASYMMETRY * s_asym +
                               W_BLINK * s_blink +
                               W_GAZE * s_gaze +
                               W_SPIKE * s_spike)

                iris_history.append(s_iris)
                micro_history.append(s_micro)
                asymmetry_history.append(s_asym)
                blink_history.append(s_blink)
                gaze_history.append(s_gaze)
                verdict_history.append(verdict_val)

                if verdict_val > COMBINED_THRESHOLD:
                    status_text = "!! SUSPICIOUS !!"
                    status_color = (0, 0, 255)
                elif verdict_val > COMBINED_THRESHOLD * 0.6:
                    status_text = "SLIGHTLY NERVOUS"
                    status_color = (0, 165, 255)
                else:
                    status_text = "TRUTHFUL"
                    status_color = (0, 255, 0)
        else:
            if mode == "INTERROGATION":
                status_text = "BLINK"
                status_color = (180, 180, 180)

    # ====================== UI DRAWING ======================
    overlay = frame.copy()

    # -- Top bar --
    cv2.rectangle(overlay, (0, 0), (iw, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, status_text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 3)

    # -- Mode badge --
    mode_color = (255, 200, 0) if mode == "LEARNING" else (0, 200, 255)
    cv2.putText(frame, mode, (iw - 260, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    # -- Gauges (right side) --
    if mode == "INTERROGATION" and locked:
        gx, gy = iw - 260, 95
        gauge_data = [
            (iris_history,      "Iris",      (200, 150, 0)),
            (micro_history,     "Micro",     (0, 150, 200)),
            (asymmetry_history, "Asymmetry", (180, 0, 180)),
            (blink_history,     "Blink Rate",(0, 200, 150)),
            (gaze_history,      "Gaze",      (150, 150, 0)),
            (verdict_history,   "VERDICT",   None),
        ]
        for i, (hist, label, color) in enumerate(gauge_data):
            if len(hist) > 0:
                v = hist[-1]
                if color is None:
                    color = (0, 0, 255) if v > COMBINED_THRESHOLD else (0, 200, 0)
                draw_gauge(frame, gx, gy + i * 30, 240, 16, v, label, color)

    # -- Rolling graph at bottom --
    graph_h = 90
    graph_y = ih - graph_h - 10
    cv2.rectangle(frame, (10, graph_y), (iw - 10, ih - 10), (30, 30, 30), -1)

    def draw_graph_line(history_buf, color):
        if len(history_buf) < 2:
            return
        graph_w = iw - 20
        step = graph_w / max(len(history_buf) - 1, 1)
        pts = []
        for i, v in enumerate(history_buf):
            x = int(10 + i * step)
            y = int(graph_y + graph_h - np.clip(v, 0, 1) * graph_h)
            pts.append([x, y])
        cv2.polylines(frame, [np.array(pts, np.int32)], False, color, 1)

    if mode == "INTERROGATION":
        draw_graph_line(iris_history,      (200, 150, 0))
        draw_graph_line(asymmetry_history, (180, 0, 180))
        draw_graph_line(blink_history,     (0, 200, 150))
        draw_graph_line(verdict_history,   (255, 255, 255))
        # Threshold line
        ty = int(graph_y + graph_h - COMBINED_THRESHOLD * graph_h)
        cv2.line(frame, (10, ty), (iw - 10, ty), (0, 0, 180), 1)

    # Legend
    labels = [("Iris",(200,150,0)), ("Asym",(180,0,180)),
              ("Blink",(0,200,150)), ("Verdict",(255,255,255))]
    for i, (lbl, c) in enumerate(labels):
        cv2.putText(frame, lbl, (15 + i * 65, graph_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1)

    cv2.imshow('MakeUofT AI Polygraph', frame)

    # ====================== KEY HANDLING ======================
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if mode == "LEARNING" and len(baseline_iris) >= 10:
            # Compute baseline blink rate (blinks per minute)
            if len(baseline_blink_ts) >= 2:
                bl_duration = baseline_blink_ts[-1] - baseline_blink_ts[0]
                bl_rate = len(baseline_blink_ts) / max(bl_duration, 1e-6) * 60.0
            else:
                bl_rate = 15.0  # average human blink rate as fallback

            locked = {
                'iris_m':     np.mean(baseline_iris),
                'iris_s':     max(np.std(baseline_iris), 1e-6),
                'brightness': np.mean(baseline_brightness),
                'micro_m':    np.mean(baseline_micro),
                'micro_s':    max(np.std(baseline_micro), 1e-6),
                'asym_m':     np.mean(baseline_asymmetry),
                'asym_s':     max(np.std(baseline_asymmetry), 1e-6),
                'gaze_m':     np.mean(baseline_gaze),
                'gaze_s':     max(np.std(baseline_gaze), 1e-6),
                'blink_rate': bl_rate,
            }
            mode = "INTERROGATION"
            for h in [iris_history, micro_history, asymmetry_history,
                      blink_history, gaze_history, verdict_history]:
                h.clear()
            print(f"[LOCKED] iris={locked['iris_m']:.5f}+-{locked['iris_s']:.5f}  "
                  f"blink={locked['blink_rate']:.1f}/min  "
                  f"asym={locked['asym_m']:.4f}")

        elif mode == "INTERROGATION":
            mode = "LEARNING"
            for buf in [baseline_iris, baseline_brightness, baseline_micro,
                        baseline_asymmetry, baseline_gaze, baseline_blink_ts,
                        rolling_iris, rolling_micro, rolling_asymmetry,
                        rolling_gaze, micro_raw_buf, blink_timestamps]:
                buf.clear()
            locked = {}
            print("[RESET] Back to LEARNING mode")

    elif key == ord('r'):
        mode = "LEARNING"
        for buf in [baseline_iris, baseline_brightness, baseline_micro,
                    baseline_asymmetry, baseline_gaze, baseline_blink_ts,
                    rolling_iris, rolling_micro, rolling_asymmetry,
                    rolling_gaze, micro_raw_buf, blink_timestamps,
                    iris_history, micro_history, asymmetry_history,
                    blink_history, gaze_history, verdict_history]:
            buf.clear()
        locked = {}
        was_blinking = False
        print("[FULL RESET] Cleared everything")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()