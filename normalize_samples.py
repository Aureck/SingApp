import os, cv2, numpy as np, shutil
from constants import FRAME_ACTIONS_PATH, MODEL_FRAMES

def read_frames(dir_):
    frames = []
    for fname in sorted(os.listdir(dir_)):
        if fname.lower().endswith(".jpg"):
            f = cv2.imread(os.path.join(dir_, fname))
            if f is not None: frames.append(f)
    return frames

def _interp_frames(frames, target=MODEL_FRAMES):
    cur = len(frames)
    if cur == target: return frames
    idxs = np.linspace(0, cur-1, target)
    out = []
    for i in idxs:
        lo, hi = int(np.floor(i)), int(np.ceil(i))
        w = i - lo
        if lo == hi: out.append(frames[lo])
        else:
            out.append(cv2.addWeighted(frames[lo], 1-w, frames[hi], w, 0))
    return out

def _downsample(frames, target=MODEL_FRAMES):
    cur = len(frames)
    step = cur / target
    idxs = np.arange(0, cur, step).astype(int)[:target]
    return [frames[i] for i in idxs]

def normalize(frames, target=MODEL_FRAMES):
    if len(frames) < target:  return _interp_frames(frames, target)
    if len(frames) > target:  return _downsample(frames, target)
    return frames

def clear_and_save(dir_, frames):
    for f in os.listdir(dir_):
        p = os.path.join(dir_, f)
        if os.path.isfile(p): os.remove(p)
        else: shutil.rmtree(p)
    for i, fr in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(dir_, f"{i:02}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])

if __name__ == "__main__":
    for word in os.listdir(FRAME_ACTIONS_PATH):
        wdir = os.path.join(FRAME_ACTIONS_PATH, word)
        if not os.path.isdir(wdir): continue
        print(f"[NORMALIZE] {word}")
        for sample in os.listdir(wdir):
            sdir = os.path.join(wdir, sample)
            if not os.path.isdir(sdir): continue
            frames = read_frames(sdir)
            if not frames: continue
            out = normalize(frames, MODEL_FRAMES)
            clear_and_save(sdir, out)
