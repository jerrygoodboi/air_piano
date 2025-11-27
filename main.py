"""
Air Piano (one octave) using MediaPipe fingertip control.

- 12 notes: C4 -> B4 including sharps (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- Displays a nice piano UI and Camera Debug window
- Plays synthesized sine tones using simpleaudio
"""

import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import math
import time

# ---------------- CONFIG ----------------
CAM_SRC = "https://10.9.20.154:8080/video"
#CAM_SRC = 0  # 0 for local webcam, or "http://ip:port/video"
SAMPLE_RATE = 44100
NOTE_DURATION = 0.8  # seconds for buffer (played once)
ATTACK = 0.01
RELEASE = 0.12
WINDOW_NAME = "Air Piano"
DEBUG_WINDOW = "Camera Debug"
# ----------------------------------------

# ---------------- MediaPipe init ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    model_complexity=1
)

# ---------------- Audio: generate note buffers ----------------
# Frequencies for C4..B4 semitones:
# C4 = 261.6256 Hz, then semitone steps (multiply by 2^(n/12))
C4 = 261.6255653005986
semitone = lambda n: C4 * (2 ** (n / 12.0))

# list of semitone offsets C to B (C=0 ... B=11)
note_offsets = list(range(12))
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
frequencies = [semitone(n) for n in note_offsets]

def make_tone(freq, duration=NOTE_DURATION, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), False)
    # sine tone
    tone = np.sin(2 * np.pi * freq * t)
    # apply amplitude envelope (attack-release)
    env = np.ones_like(tone)
    a_len = int(sr * ATTACK)
    r_len = int(sr * RELEASE)
    if a_len > 0:
        env[:a_len] = np.linspace(0.0, 1.0, a_len)
    if r_len > 0:
        env[-r_len:] = np.linspace(1.0, 0.0, r_len)
    tone = tone * env
    # scale to 16-bit PCM
    audio = (tone * 32767 * 0.7).astype(np.int16)
    return audio.tobytes()

# Pre-generate audio buffers and WaveObjects
note_wave_objs = []
for f in frequencies:
    buf = make_tone(f)
    note_wave_objs.append(sa.WaveObject(buf, 1, 2, SAMPLE_RATE))


# ---------------- UI geometry helpers ----------------
# We'll layout seven white keys across the display width, then place black keys above them.
def layout_keys(frame_w, frame_h, bottom_margin=60):
    """Return lists of white keys and black keys with their rects and associated note indices."""
    # piano width margin
    left = int(frame_w * 0.06)
    right = int(frame_w * 0.94)
    width = right - left
    white_count = 7
    white_w = width / white_count
    white_h = int(frame_h * 0.45)  # vertical size of keys
    top = frame_h - white_h - bottom_margin

    # white key rectangles and mapping to semitone index
    # Order of white keys from left: C D E F G A B
    white_order = [0, 2, 4, 5, 7, 9, 11]  # semitone offsets for whites (relative to C)
    whites = []
    for i, sem in enumerate(white_order):
        x1 = int(left + i * white_w)
        x2 = int(left + (i + 1) * white_w)
        rect = (x1, top, x2, top + white_h)  # (x1,y1,x2,y2)
        whites.append({"rect": rect, "semitone": sem, "i": i})

    # black keys sit between whites; skip between E-F and B-C (i.e. no black after E (index 2) and after B (index 6))
    # black key width ~ 60% of white width, height ~ 60%
    black_w = int(white_w * 0.62)
    black_h = int(white_h * 0.62)
    blacks = []
    # position black keys between certain white keys: between C-D, D-E, F-G, G-A, A-B => indexes 0,1,3,4,5
    black_pairs = [0, 1, 3, 4, 5]  # white indices to the left of black key
    # corresponding semitone offsets for those black keys: C#=1, D#=3, F#=6, G#=8, A#=10
    black_semitones = [1, 3, 6, 8, 10]
    for j, left_white_idx in enumerate(black_pairs):
        left_rect = whites[left_white_idx]["rect"]
        # center between left white and next white
        x1_left = left_rect[0]
        x2_right = whites[left_white_idx + 1]["rect"][2]
        cx = int((x1_left + x2_right) / 2)
        bx1 = cx - black_w // 2
        bx2 = bx1 + black_w
        by1 = top
        by2 = top + black_h
        rect = (bx1, by1, bx2, by2)
        blacks.append({"rect": rect, "semitone": black_semitones[j], "i": j})

    return whites, blacks

# ---------------- Press detection: map x,y to semitone ----------------
def point_in_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return (x >= x1 and x <= x2 and y >= y1 and y <= y2)

def hit_test(pt, whites, blacks):
    # check black keys top-down first (they visually overlay whites)
    for b in blacks:
        if point_in_rect(pt, b["rect"]):
            return b["semitone"], 'black', b
    for w in whites:
        if point_in_rect(pt, w["rect"]):
            return w["semitone"], 'white', w
    return None, None, None

# ---------------- Main loop ----------------
cap = cv2.VideoCapture(CAM_SRC)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Cannot open camera. Change CAM_SRC to your camera or IP stream.")

H, W = frame.shape[:2]

# precompute layout
whites, blacks = layout_keys(W, H)

# press state: dict semitone -> bool
pressed_state = {}
for n in range(12):
    pressed_state[n] = False

# last play timestamp to avoid immediate re-trigger if finger jittered
last_play_time = {n: 0.0 for n in range(12)}
min_note_gap = 0.12  # seconds between retriggers of the same key

print("Air Piano ready. Press ESC to quit.")

while True:
    ret, cam = cap.read()
    if not ret:
        break
    cam = cv2.flip(cam, 1)
    frame = np.ones_like(cam) * 245  # light background (off-white)

    # optional: draw subtle piano background shadow/board
    board_h = int(H * 0.58)
    board_w = int(W * 0.88)
    board_x = int(W * 0.06)
    board_y = int(H * 0.15)
    cv2.rectangle(frame, (board_x, board_y), (board_x + board_w, board_y + board_h), (230, 230, 230), -1)
    cv2.rectangle(frame, (board_x, board_y), (board_x + board_w, board_y + board_h), (200, 200, 200), 2)

    # draw white keys
    for wkey in whites:
        x1, y1, x2, y2 = wkey["rect"]
        # key base
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # light border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        # label
        cx = int((x1 + x2) / 2)
        label = note_names[wkey["semitone"]]
        cv2.putText(frame, label, (cx - 14, y2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    # draw black keys
    for bkey in blacks:
        x1, y1, x2, y2 = bkey["rect"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        # slight highlight top
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

    # Hand detection
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark[8]  # index tip
        fx, fy = int(lm.x * W), int(lm.y * H)
        fingertip = (fx, fy)
        # draw fingertip dot (bigger and glowing)
        cv2.circle(frame, fingertip, 14, (0, 0, 210), -1)
        cv2.circle(frame, fingertip, 20, (100, 100, 255), 1)

    # If fingertip present, test for key hit
    if fingertip:
        sem, kind, meta = hit_test(fingertip, whites, blacks)
        if sem is not None:
            # visual highlight
            if kind == 'black':
                x1, y1, x2, y2 = meta["rect"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 180), -1)
            else:
                x1, y1, x2, y2 = meta["rect"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 235, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # press detection & sound trigger: finger must be in lower half of key to count as press
            y_threshold = int((y1 + y2) * 0.55) if kind == 'white' else int((y1 + y2) * 0.7)
            if fingertip[1] > y_threshold:
                now = time.time()
                # simple debounce
                if not pressed_state[sem] and now - last_play_time[sem] > min_note_gap:
                    # play sound
                    try:
                        note_wave_objs[sem].play()
                    except Exception:
                        pass
                    pressed_state[sem] = True
                    last_play_time[sem] = now
            else:
                # hovering but not pressed: do not change pressed state
                pass
        else:
            # fingertip outside keys: clear any pressed states (so new presses allowed when enter)
            for k in pressed_state:
                pressed_state[k] = False
    else:
        # no fingertip detected, clear pressed states so next detection triggers clean press
        for k in pressed_state:
            pressed_state[k] = False

    # visual pressed overlay for pressed keys
    for sem_idx, pressed in pressed_state.items():
        if pressed:
            # find if sem corresponds to black or white for overlay
            # black semitones set = {1,3,6,8,10}
            if sem_idx in {1, 3, 6, 8, 10}:
                # find black by semitone
                for b in blacks:
                    if b["semitone"] == sem_idx:
                        x1, y1, x2, y2 = b["rect"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 160), -1)
            else:
                for w in whites:
                    if w["semitone"] == sem_idx:
                        x1, y1, x2, y2 = w["rect"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 230, 255), -1)

    # draw labels title
    cv2.putText(frame, "Air Piano — C4 to B4 (touch lower area to play)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)

    # show camera debug window (small, on the side)
    # We'll show the raw camera separately so you can monitor tracking
    cv2.imshow(DEBUG_WINDOW, cam)
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# cleanup
cap.release()
cv2.destroyAllWindows()

