import os, csv, cv2
import numpy as np
from glob import glob
from mediapipe.python.solutions.hands import Hands

from constants import (
    FRAME_ACTIONS_PATH, CSV_PATH, PROC_SIZE, LABEL_COL,
    AUG_PER_FRAME, AUG_ROT_MAX_DEG, AUG_SCALE_MINMAX, AUG_JITTER_STD
)
from helpers import ensure_dir, normalized_twohands_vector
from augment import augment_vec   

def _csv_header():
    cols=[]
    for i in range(21): cols += [f"Lx{i}", f"Ly{i}", f"Lz{i}"]
    for i in range(21): cols += [f"Rx{i}", f"Ry{i}", f"Rz{i}"]
    cols.append(LABEL_COL)
    return cols

def process_videos_to_csv():
    ensure_dir(os.path.dirname(CSV_PATH))
    newfile = (not os.path.exists(CSV_PATH)) or os.stat(CSV_PATH).st_size == 0
    if newfile:
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_csv_header())

    hands = Hands(static_image_mode=False, max_num_hands=2,
                  min_detection_confidence=0.6, min_tracking_confidence=0.5)

    total = 0
    for label in next(os.walk(FRAME_ACTIONS_PATH))[1]:
        folder = os.path.join(FRAME_ACTIONS_PATH, label)
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for video in sorted(glob(os.path.join(folder, ext))):
                mark = video + ".done"
                if os.path.exists(mark) and os.path.getmtime(mark) >= os.path.getmtime(video):
                    continue

                cap = cv2.VideoCapture(video)
                if not cap.isOpened(): continue
                rows = 0
                with open(CSV_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    while True:
                        ok, frame = cap.read()
                        if not ok: break
                        small = cv2.resize(frame, PROC_SIZE)
                        res = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                        if res.multi_hand_landmarks:
                            vec = normalized_twohands_vector(res)
                            w.writerow(list(vec) + [label])
                            rows += 1
                            # ------- AUGMENTATION -------
                            for _ in range(int(AUG_PER_FRAME)):
                                avec = augment_vec(
                                    vec,
                                    rot_deg=AUG_ROT_MAX_DEG,
                                    scale_minmax=AUG_SCALE_MINMAX,
                                    jitter_std=AUG_JITTER_STD
                                )
                                w.writerow(list(avec) + [label])
                                rows += 1
                cap.release()
                open(mark, "w").close()
                print(f"[VID] {os.path.basename(video)} -> {rows} filas (+aug={AUG_PER_FRAME}x)")
                total += rows
    hands.close()
    print(f"[CSV] Total filas nuevas: {total}")

if __name__ == "__main__":
    process_videos_to_csv()
