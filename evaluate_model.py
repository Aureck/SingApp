import os, cv2, numpy as np
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, draw_keypoints, there_hand, get_word_ids, extract_keypoints
from constants import (
    KERAS_MODEL_PATH, WORDS_JSON_PATH, MODEL_FRAMES, THRESHOLD_PRED,
    MIN_LENGTH_FRAMES, FONT, FONT_POS, FONT_SIZE
)

def _interp_seq(seq, target=MODEL_FRAMES):
    cur = len(seq)
    if cur == target: return np.array(seq, dtype=np.float32)
    idxs = np.linspace(0, cur-1, target)
    out = []
    for i in idxs:
        lo, hi = int(np.floor(i)), int(np.ceil(i))
        w = i - lo
        if lo == hi: out.append(seq[lo])
        else: out.append((1-w)*seq[lo] + w*seq[hi])
    return np.array(out, dtype=np.float32)

def normalize_keypoints(seq, target=MODEL_FRAMES):
    if len(seq) < target:  return _interp_seq(seq, target)
    if len(seq) > target:
        step = len(seq)/target
        idxs = np.arange(0, len(seq), step).astype(int)[:target]
        return np.array([seq[i] for i in idxs], dtype=np.float32)
    return np.array(seq, dtype=np.float32)

def evaluate_model(src=None, threshold=THRESHOLD_PRED, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    words = get_word_ids(WORDS_JSON_PATH)  # lista como ["hola-0", "adios-1", ...] o similar
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo Keras: {KERAS_MODEL_PATH}")
    model = load_model(KERAS_MODEL_PATH)

    count, fix, recording = 0, 0, False

    with Holistic() as holistic:
        cap = cv2.VideoCapture(0 if src is None else src)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            results = mediapipe_detection(frame, holistic)

            if there_hand(results) or recording:
                recording = False
                count += 1
                if count > margin_frame:
                    kp_seq.append(extract_keypoints(results))
            else:
                if count >= MIN_LENGTH_FRAMES + margin_frame:
                    fix += 1
                    if fix < delay_frames:
                        recording = True
                    else:
                        kp_seq = kp_seq[:-(margin_frame + delay_frames)]
                        kp_norm = normalize_keypoints(kp_seq, MODEL_FRAMES)
                        res = model.predict(np.expand_dims(kp_norm, axis=0), verbose=0)[0]
                        idx = int(np.argmax(res))
                        if res[idx] >= threshold:
                            wid = words[idx].split('-')[0] if idx < len(words) else f"idx_{idx}"
                            sentence.insert(0, wid.upper())
                            print(wid, f"({res[idx]*100:.1f}%)")
                recording, fix, count, kp_seq = False, 0, 0, []

            if src is None:
                cv2.rectangle(frame, (0,0), (640,35), (0,0,0), -1)
                cv2.putText(frame, ' | '.join(sentence[:5]), FONT_POS, FONT, FONT_SIZE, (255,255,255), 1)
                draw_keypoints(frame, results)
