import os, json, cv2, time
import numpy as np
import unicodedata
from typing import Tuple, List
from mediapipe.python.solutions import hands as mp_hands, drawing_utils as mp_draw
from mediapipe.python.solutions.holistic import Holistic

from constants import META_PATH, PROC_SIZE, VIDEO_FPS, VIDEO_CODECS

# ---------- util ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def create_folder(p):  
    ensure_dir(p)

def sanitize_label(text: str) -> str:
    if not text: 
        return "sin_nombre"
    t = _strip_accents(text.strip().lower())
    for ch in '<>:"/\\|?*.,;!¡¿?':
        t = t.replace(ch, ' ')
    return "_".join([x for x in t.split() if x])

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if unicodedata.category(ch) != "Mn")

# ---------- MediaPipe (Hands) ----------
def normalized_twohands_vector(res) -> List[float]:
    """
    Vector 126 floats = 63 (mano izq) + 63 (mano der).
    Normaliza centrando en muñeca y escalando por palma.
    """
    def norm21(landmarks):
        pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)  # (21,3) 0..1
        center = pts[0]
        palm = pts[[5,9,13,17]].mean(axis=0)
        scale = float(np.linalg.norm(palm - center) + 1e-6)
        pts = (pts - center) / scale
        return pts

    L = np.zeros((21,3), dtype=np.float32)
    R = np.zeros((21,3), dtype=np.float32)

    if res.multi_hand_landmarks:
        handed = []
        if getattr(res, "multi_handedness", None):
            handed = [h.classification[0].label for h in res.multi_handedness]

        for i, hand in enumerate(res.multi_hand_landmarks[:2]):
            lab = handed[i] if i < len(handed) else ("Left" if i==0 else "Right")
            pts = norm21(hand.landmark)
            if lab.lower().startswith("left"): L = pts
            else: R = pts

    return list(L.flatten()) + list(R.flatten())  # 126

def draw_hands(frame_bgr, res):
    if not res or not res.multi_hand_landmarks: return
    for h in res.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame_bgr, h, mp_hands.HAND_CONNECTIONS)

# ---------- Holistic helpers (para Keras y captura de frames) ----------
def mediapipe_detection(image_bgr, holistic: Holistic):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    return res

def there_hand(results) -> bool:
    return bool(getattr(results, 'left_hand_landmarks', None) or
                getattr(results, 'right_hand_landmarks', None))

def draw_keypoints(image_bgr, results):
    if getattr(results, 'left_hand_landmarks', None):
        for lm in results.left_hand_landmarks.landmark:
            cx, cy = int(lm.x * image_bgr.shape[1]), int(lm.y * image_bgr.shape[0])
            cv2.circle(image_bgr, (cx, cy), 2, (0, 255, 0), -1)
    if getattr(results, 'right_hand_landmarks', None):
        for lm in results.right_hand_landmarks.landmark:
            cx, cy = int(lm.x * image_bgr.shape[1]), int(lm.y * image_bgr.shape[0])
            cv2.circle(image_bgr, (cx, cy), 2, (0, 180, 255), -1)

def extract_keypoints(results):
    def _hand21(lm):
        if lm is None: return [0.0]*63
        pts = []
        for p in lm.landmark:
            pts += [p.x, p.y, p.z]
        return pts
    L = _hand21(getattr(results, 'left_hand_landmarks', None))
    R = _hand21(getattr(results, 'right_hand_landmarks', None))
    return L + R  
# ---------- face helpers ----------
def extract_face(results, max_points: int = 50):
    """
    Convierte landmarks de la cara en un vector de floats.
    Usamos solo los primeros max_points para no hacerlo gigante.
    Cada punto tiene (x, y, z).
    """
    if not getattr(results, 'face_landmarks', None):
        return [0.0] * (max_points * 3)
    pts = []
    for lm in results.face_landmarks.landmark[:max_points]:
        pts += [lm.x, lm.y, lm.z]
    while len(pts) < max_points * 3:
        pts.append(0.0)
    return pts

def extract_hands_and_face(results, face_points: int = 50):
    """
    Devuelve un vector concatenado [manos (126) + cara (face_points*3)].
    """
    hands_vec = extract_keypoints(results)  # 126 floats
    face_vec  = extract_face(results, face_points)
    return hands_vec + face_vec

# ---------- words.json (Keras opcional) ----------
def get_word_ids(json_path):
    if not os.path.exists(json_path): return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- model meta ----------
def save_meta(meta: dict):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta() -> dict:
    if not os.path.exists(META_PATH): return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- cámara / video ----------
def open_camera_any() -> cv2.VideoCapture | None:
    for idx in (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    for idx in (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    for idx in (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): return cap
    return None

def prepare_writer(out_folder:str, base:str, size:Tuple[int,int]):
    ensure_dir(out_folder)
    for ext, fourcc_name in VIDEO_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        path = os.path.join(out_folder, f"{base}.{ext}")
        wr = cv2.VideoWriter(path, fourcc, VIDEO_FPS, size)
        if wr.isOpened(): return wr, path
    return None, None

# ---------- rutas de datos para PyInstaller ----------
import sys

def get_data_path(relative_path: str) -> str:
    """
    Devuelve la ruta absoluta a un recurso,
    tanto en desarrollo como dentro del exe empaquetado.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
