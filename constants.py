import os
import cv2

# ======== ROOT & PATHS ========
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FRAME_ACTIONS_PATH, exist_ok=True)
os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)

# Archivos de datos
DATA_JSON_PATH  = os.path.join(DATA_PATH, "data.json")
META_PATH       = os.path.join(MODEL_FOLDER_PATH, "model_meta.json")
CSV_PATH        = os.path.join(DATA_PATH, "keypoints.csv")   

# ======== MODELOS ========
MODEL_FRAMES = 15
KERAS_MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")  
SK_MODEL_PATH    = os.path.join(MODEL_FOLDER_PATH, "sgd_signs.joblib")              
WORDS_JSON_PATH  = os.path.join(MODEL_FOLDER_PATH, "words.json")                     

# ======== VÍDEO/CÁMARA ========
TARGET_W, TARGET_H = 1280, 720
VIDEO_FPS = 60
VIDEO_CODECS = [("mp4", "mp4v"), ("avi", "XVID"), ("mov", "mp4v")]

# ======== DIBUJO EN PANTALLA ========
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# ======== PIPELINE KERAS ========
MIN_LENGTH_FRAMES = 5
THRESHOLD_PRED = 0.7
PROC_SIZE = (224, 224)   # normalización para holistic
LENGTH_KEYPOINTS = 1662 

# ======== PIPELINE SCIKIT ========
LABEL_COL = "label"
N_FEATS   = 126         
MIN_CONF  = 0.60
PROCESS_EVERY = 2        # inferir cada N frames

# ======== GRABACIÓN (UI QT) ========
RECORD_SECONDS = 6

# ======== AUTO-FLUJO ========
AUTO_TRAIN_AFTER_RECORD = True   # entrenar automáticamente tras grabar
AUTO_PROCESS_AFTER_RECORD = True # generar/actualizar CSV automáticamente

# ======== AUGMENTATION (solo para CSV de scikit) ========
AUG_PER_FRAME     = 3        # cuántas variantes por frame (0 = sin augmentation)
AUG_ROT_MAX_DEG   = 12       # rotación ± grados (en plano XY)
AUG_SCALE_MINMAX  = (0.90, 1.10)
AUG_JITTER_STD    = 0.02     # ruido gaussiano (std) sobre x,y,z normalizados

# ======== INFERENCIA EN TIEMPO REAL ========
MIN_CONF          = 0.55     # baja si no detecta (0.45–0.6)
PROCESS_EVERY     = 4        # frames entre inferencias (3-6)
STABLE_WINDOW     = 8        # historial de predicciones
STABLE_REQUIRE    = 4        # mínimo repeticiones del top label dentro del historial
