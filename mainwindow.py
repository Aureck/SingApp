import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys, time, cv2, re, unicodedata, glob
import numpy as np
from datetime import datetime
from collections import deque
from PyQt5.QtGui import QColor

from joblib import load
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QSplitter,
    QInputDialog, QMessageBox, QCheckBox, QDialog, QSlider, QProgressDialog,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QSizePolicy




from mediapipe.python.solutions.hands import Hands

# === Constantes del proyecto ===
from constants import (
    TARGET_W, TARGET_H, FRAME_ACTIONS_PATH, RECORD_SECONDS,
    PROCESS_EVERY, MIN_CONF, SK_MODEL_PATH, PROC_SIZE,
    AUTO_PROCESS_AFTER_RECORD, AUTO_TRAIN_AFTER_RECORD,
    STABLE_WINDOW, STABLE_REQUIRE, DATA_PATH
)

# === Helpers del proyecto ===
from helpers import (
    ensure_dir, sanitize_label, draw_hands, normalized_twohands_vector,
    open_camera_any, prepare_writer, load_meta
)

# Carpeta para videos de “Respuestas rápidas”
QUICK_VIDEOS_PATH = os.path.join(DATA_PATH, "quick_videos")
os.makedirs(QUICK_VIDEOS_PATH, exist_ok=True)


# ============================================================
# Utilidades de texto y matching
# ============================================================
def _strip_accents(s: str) -> str:
    if not isinstance(s, str): return ""
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                    if unicodedata.category(ch) != "Mn")

def _clean_text_to_words(s: str) -> list:
    # minuscula, sin acentos, quita signos, separa por espacios
    s = _strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []

def _possible_label_variants(words: list) -> list:
    """
    Dado un grupo de palabras contiguas, devuelve variantes de label:
    ["buenos_dias", "buenosdias", "buenos-dias", "buenos dias"]
    """
    base = " ".join(words)
    no_space = base.replace(" ", "")
    with_us = "_".join(words)
    with_dash = "-".join(words)
    return [with_us, no_space, with_dash, base]


# ============================================================
# Worker de Background (procesar CSV y entrenar)
# ============================================================
class ProcTrainWorker(QThread):
    finished = pyqtSignal(bool, str)  

    def __init__(self, do_process=True, do_train=True, parent=None):
        super().__init__(parent)
        self.do_process = do_process
        self.do_train = do_train

    def run(self):
        try:
            if self.do_process:
                # aquí tu lógica de procesamiento CSV
                pass
            if self.do_train:
                # aquí tu lógica de entrenamiento
                pass
            self.finished.emit(True, "Entrenamiento finalizado correctamente")
        except Exception as e:
            self.finished.emit(False, str(e))


# ============================================================
# Ventana de Grabación (separada)
# ============================================================
class RecorderWindow(QDialog):
    """
    - Preview con landmarks.
    - Inicia grabación al detectar ≥1 mano (o 2 manos si marcas el checkbox).
    - Detiene tras RECORD_SECONDS y guarda en destino (frame_actions/<label>/ o quick_videos/).
    """
    def __init__(self, parent, label: str, dest_folder: str = None):
        super().__init__(parent)
        self.setWindowTitle(f"Grabar seña: {label}")
        self.setModal(True)
        self.resize(960, 620)

        self.label = label
        self.dest_folder = dest_folder
        self.recording = False
        self.writer = None
        self.out_path = None
        self.t_end = None

        # ---------- UI: preview sin zoom (letterbox) ----------
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black")
        self.video_label.setScaledContents(False)
        base_w = 920
        base_h = int(base_w * (TARGET_H / TARGET_W))
        self.video_label.setMinimumSize(base_w, base_h)

        self.status_lbl = QLabel("Estado: Esperando manos… (≥1)")
        self.status_lbl.setStyleSheet("padding:6px; background:#eef;")
        self.timer_lbl = QLabel("Tiempo restante: -")
        self.timer_lbl.setAlignment(Qt.AlignCenter)
        self.timer_lbl.setStyleSheet("font-size:18px; font-weight:bold; color:red; padding:6px;")


        self.chk_both = QCheckBox("Requerir ambas manos para iniciar")
        self.chk_both.stateChanged.connect(self._on_toggle_both)

        self.btn_stop = QPushButton("Detener")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self._on_cancel)

        btns = QHBoxLayout()
        btns.addWidget(self.chk_both)
        btns.addStretch(1)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(self.video_label)
        lay.addWidget(self.status_lbl)
        lay.addWidget(self.timer_lbl) 
        lay.addLayout(btns)

        # ---------- Cámara + MediaPipe ----------
        self.cap = open_camera_any()
        if not self.cap:
            QMessageBox.critical(self, "Cámara", "No se pudo abrir la cámara.")
            self.reject()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

        self.hands = Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.60, min_tracking_confidence=0.50)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._loop)
        self.timer.start(30)

        self._hands_ok_frames = 0
        self._need_consecutive = 5

    # ---------- Handlers ----------
    def _on_toggle_both(self, _state):
        self._set_status("Esperando manos… (2)" if self.chk_both.isChecked() else "Esperando manos… (≥1)")

    def _on_cancel(self):
        self._stop_recording(save=False)
        self.close()

    def _on_stop(self):
        self._stop_recording(save=True)
        self.accept()

    # ---------- Loop principal ----------
    def _loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)

        # --- duplicados ---
        clean_frame = frame.copy()
        frame_draw = frame.copy()

        # --- detección de manos ---
        small = cv2.resize(frame, PROC_SIZE)
        res = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        hand_count = len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0

        # --- dibujar landmarks solo si está habilitado en draw_landmarks ---
        if res.multi_hand_landmarks and self.draw_landmarks:
            res_big = self.hands.process(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))
            draw_hands(frame_draw, res_big)

        # --- lógica de inicio/detención dinámica ---
        trigger = (hand_count >= 2) if self.chk_both.isChecked() else (hand_count >= 1)

        if not self.recording:
            if trigger:
                self._hands_ok_frames += 1
                if self._hands_ok_frames >= self._need_consecutive:
                    self._start_recording(frame.shape[1], frame.shape[0])
            else:
                self._hands_ok_frames = 0
        else:
            # guardamos ambos streams
            if self.writer:
                self.writer.write(frame_draw)
            if self.writer_clean:
                self.writer_clean.write(clean_frame)

            # detener automáticamente si ya no hay manos por varios frames
            if hand_count == 0:
                self._hands_ok_frames += 1
            else:
                self._hands_ok_frames = 0

            # si pasaron 30 frames (~1 s) sin manos, parar
            if self._hands_ok_frames >= 30:
                self._stop_recording(save=True)
                self.accept()

            # actualizar estado
            self.timer_lbl.setText("⏱ Grabando... (se detendrá al quitar las manos)")

        # mostrar
        self._blit_to_label(frame_draw)




    # ---------- Grabación ----------
    def _start_recording(self, W, H):
        folder = self.dest_folder if self.dest_folder else os.path.join(FRAME_ACTIONS_PATH, self.label)
        ensure_dir(folder)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Archivo con puntos
        self.writer, self.out_path = prepare_writer(
            folder, f"{self.label}_{ts}", (W, H)
        )

        # Archivo sin puntos (extra)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clean_path = os.path.join(folder, f"{self.label}_{ts}_clean.mp4")
        self.writer_clean = cv2.VideoWriter(clean_path, fourcc, 20.0, (W, H))
        self.out_path_clean = clean_path

        if not self.writer or not self.writer_clean.isOpened():
            QMessageBox.critical(self, "Video", "No fue posible iniciar el video writer.")
            self._set_status("ERROR al iniciar grabación")
            return

        self.recording = True
        self.t_end = time.time() + RECORD_SECONDS
        self.btn_stop.setEnabled(True)
        self._set_status(f"Grabando… ({'2 manos' if self.chk_both.isChecked() else '≥1 mano'})")


    def _stop_recording(self, save=True):
        if self.writer:
            try:
                self.writer.release()
            except:
                pass

        if self.writer_clean:
            try:
                self.writer_clean.release()
            except:
                pass

        # Guardar solo una versión limpia "oficial" por etiqueta
        if save and self.out_path:
            clean_global = os.path.join(self.dest_folder or FRAME_ACTIONS_PATH, f"{self.label}_clean.mp4")
            if not os.path.exists(clean_global):
                try:
                    import shutil
                    shutil.copy(self.out_path_clean, clean_global)
                    print(f"[REC] Copia limpia guardada como: {clean_global}")
                except Exception as e:
                    print("[REC] Error copiando limpio:", e)

            QMessageBox.information(
                self, "Grabación",
                f"Videos guardados:\n{self.out_path}\n{self.out_path_clean}"
            )

        self.writer = None
        self.writer_clean = None
        self.recording = False
        self._hands_ok_frames = 0
        self.t_end = None
        self.btn_stop.setEnabled(False)
        self._set_status("Listo. (Puedes cerrar esta ventana)")



    # ---------- Utils ----------
    def _set_status(self, text):
        self.status_lbl.setText(f"Estado: {text}")

    def _blit_to_label(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        pm = pm.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pm)
    

    def closeEvent(self, e):
        try:
            if self.timer and self.timer.isActive():
                self.timer.stop()
        except:
            pass
        try:
            self.hands.close()
        except:
            pass
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except:
            pass
        super().closeEvent(e)


# ============================================================
# Ventana de Reproducción (pantalla completa)
# ============================================================
from PyQt5.QtGui import QGuiApplication

class PlayerWindow(QDialog):
    def __init__(self, parent, video_path: str, model=None, meta=None, tokens_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Reproducción")
        self.setModal(True)
        self.video_path = video_path

        self.model = model
        self.meta = meta
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.resize(800, 600)

        # UI fullscreen
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black")
        self.video_label.setScaledContents(False)

        self.bottom_lbl = QLabel(tokens_text)
        self.bottom_lbl.setAlignment(Qt.AlignCenter)
        self.bottom_lbl.setStyleSheet("font-size:22px; padding:10px; background:#111; color:white;")

        lay = QVBoxLayout(self)
        lay.addWidget(self.video_label, stretch=1)
        lay.addWidget(self.bottom_lbl)

        # Media
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Video", "No se pudo abrir el video.")
            self.reject(); return

        self.hands = Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.60, min_tracking_confidence=0.50)

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._loop)
        self.timer.start(max(10, int(1000.0 / fps)))

        # --- 🔹 Detectar pantallas disponibles ---
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            # Pantalla secundaria
            second_screen = screens[1]
            geo = second_screen.geometry()
            self.setGeometry(geo)  # mueve la ventana completa
        else:
            # Si solo hay 1 pantalla, usa fullscreen normal
            self.showFullScreen()

    
    def _loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.close()  
            return
        # detección
        small = cv2.resize(frame, PROC_SIZE)
        res = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            res_big = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_hands(frame, res_big)

        # predicción opcional
        if self.model is not None:
            try:
                scaler = self.model["scaler"]; clf = self.model["clf"]
                vec = normalized_twohands_vector(res) if res and res.multi_hand_landmarks else None
                if vec is not None:
                    Xs = scaler.transform([np.array(vec, dtype=np.float32)])
                    proba = clf.predict_proba(Xs)[0]
                    idx = int(np.argmax(proba))
                    pred = str(clf.classes_[idx]); conf = float(proba[idx])
                    cv2.rectangle(frame, (0,0), (420,40), (0,0,0), -1)
                    cv2.putText(frame, f"{pred}  ({conf*100:.1f}%)", (10,28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            except Exception:
                pass  # silencioso

        # blit
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        pm = pm.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pm)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Q):
            self.close()

    def closeEvent(self, e):
        try:
            if self.timer and self.timer.isActive():
                self.timer.stop()
        except:
            pass
        try: self.hands.close()
        except: pass
        try:
            if self.cap:
                self.cap.release()
        except: pass
        super().closeEvent(e)


# ============================================================
# Ventana Principal (Dashboard)
# ============================================================
class LSPDashboard(QMainWindow):
    def __init__(self, role="user", section_title=None):  
        super().__init__()
        self.role = role
        self.setGeometry(80, 50, 1280, 760)

        base_title = "Reconocimiento de Lengua de Señas"
        if role == "admin":
            base_title += " (admin)"
        if section_title:
            base_title += f" - {section_title}"

        self.setWindowTitle(base_title)

        self.quick = []

        # ===== Estado de predicción y configuración =====
        self.detected_word = "..."
        self.sentence = []                  
        self.autofill_sentence = True        
        self.last_detected_time = time.time()
        self.proc_every = PROCESS_EVERY
        self.draw_landmarks = True  
        self.hist = deque(maxlen=STABLE_WINDOW)

        # Cámara principal
        self.cap = None
        self.hands = None
        self._open_main_camera()

        # Modelo
        self.model = None
        self.meta = load_meta()
        if os.path.exists(SK_MODEL_PATH):
            try:
                self.model = load(SK_MODEL_PATH)
            except Exception as e:
                print("[WARN] No se pudo cargar modelo:", e)

        # --- Respuestas rápidas y compositor ---
        self.quick_path = os.path.join(DATA_PATH, "quick_replies.json")
        self.quick = self._load_quick_replies()
        self.last_recorded_video = None
        self.compose_video_path = None

        # Cache de labels disponibles -> videos
        self.labels_index = self._build_labels_index()

        # ===== Crear pestañas =====
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_datos = self._tab_datos()
        self.tab_responder = self._tab_responder()
        self.tab_settings = self._tab_settings()
        self.tab_history = self._tab_history()

        self.tabs.addTab(self.tab_datos, "Datos")
        self.tabs.addTab(self.tab_responder, "Responder")
        self.tabs.addTab(self.tab_settings, "Ajustes")
        self.tabs.addTab(self.tab_history, "Historial")

        # 🔹 Ocultar pestaña "Datos" si está en modo atención
        if section_title == "Atención a usuarios no oyentes":
            idx = self.tabs.indexOf(self.tab_datos)
            if idx != -1:
                self.tabs.removeTab(idx)


        # Control de acceso según rol
        if self.role == "usuario":
            # Eliminar directamente las pestañas que no corresponden
            for tab in [self.tab_datos, self.tab_settings, self.tab_history]:
                idx = self.tabs.indexOf(tab)
                if idx != -1:
                    self.tabs.removeTab(idx)

        # UI principal
        self._build_ui()

        # Timer de cámara
        self.frame_i = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

        # Progress dialog holder y worker
        self._progress = None
        self._worker = None


  # ---------- subir seña ----------
    


    # ---------- Abrir/Reabrir cámara principal ----------
    def _open_main_camera(self):
        # cerrar recursos previos si los había
            try:
                if self.hands:
                    self.hands.close()
            except:
                pass
            try:
                if self.cap:
                    self.cap.release()
            except:
                pass

        # abrir cámara
            self.cap = open_camera_any()
            if not self.cap:
                QMessageBox.critical(self, "Cámara", "No se pudo abrir la cámara.")
                return

            # configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

            # inicializar mediapipe Hands
            self.hands = Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.60,
                min_tracking_confidence=0.50
            )

    def ui_upload_sign(self):
        text, ok = QInputDialog.getText(self, "Nueva seña", "Nombre del label (ej. buenos_dias o hola):")
        if not ok or not text.strip():
            return
        label = sanitize_label(text)
        if not label:
            QMessageBox.warning(self, "Etiqueta", "Etiqueta inválida.")
            return

        # Elegir archivo de video
        fn, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", "", 
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not fn:
            return

        # Guardar copia en la carpeta del label
        import shutil
        folder = os.path.join(FRAME_ACTIONS_PATH, label)
        ensure_dir(folder)
        base = os.path.basename(fn)
        dest = os.path.join(folder, base)
        shutil.copy(fn, dest)

        QMessageBox.information(self, "Seña subida", f"El video fue guardado en:\n{dest}")

        # Re-indexar labels para que esté disponible
        self.labels_index = self._build_labels_index()


    def ui_record_clean(self):
        text, ok = QInputDialog.getText(self, "Grabar versión limpia", "Nombre del label (ej. hola o buenos_dias):")
        if not ok or not text.strip():
            return
        label = sanitize_label(text)
        if not label:
            QMessageBox.warning(self, "Etiqueta", "Etiqueta inválida.")
            return

        self._pause_camera(True)
        try:
            dlg = RecorderWindow(self, label)
            # 🔹 desactivar dibujo de landmarks
            dlg.hands = Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=0.60, min_tracking_confidence=0.50)
            dlg.draw_landmarks = False  # por compatibilidad
            result = dlg.exec_()
        finally:
            self._pause_camera(False)

        if result == QDialog.Accepted and getattr(dlg, "out_path_clean", None):
            # guardar solo la versión limpia como “oficial”
            import shutil
            dest = os.path.join(FRAME_ACTIONS_PATH, f"{label}_clean.mp4")
            shutil.copy(dlg.out_path_clean, dest)
            QMessageBox.information(self, "Grabación limpia", f"Versión limpia guardada:\n{dest}")
            self.labels_index = self._build_labels_index()

    # ---------- UI ----------
    def _build_ui(self):
        # ===== Vista de cámara =====
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black")
        self.video_label.setScaledContents(True)
        base_w = 960
        base_h = int(base_w * (TARGET_H / TARGET_W))
        self.video_label.setMinimumSize(base_w, base_h)

        self.lbl_detected = QLabel("Palabra detectada: ...")
        self.lbl_detected.setStyleSheet("background:#eee; font-weight:bold; padding:6px")

        self.txt_sentence = QTextEdit()
        self.txt_sentence.setPlaceholderText("Frase construida… ")
        self.txt_sentence.setReadOnly(True)
        self.txt_sentence.setStyleSheet("background:#f6fcff")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.video_label, stretch=1)
        lv.addWidget(self.lbl_detected)
        lv.addWidget(self.txt_sentence, stretch=1)

        # ===== Control de acceso =====
        if self.role == "usuario":
            # eliminar directamente las pestañas que no corresponden
            for tab in [self.tab_datos, self.tab_settings, self.tab_history]:
                idx = self.tabs.indexOf(tab)
                if idx != -1:
                    self.tabs.removeTab(idx)

        # ===== Layout general =====
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)

        # 🔹 El lado izquierdo (cámara + frase) ocupa casi todo
        splitter.setStretchFactor(0, 8)   
        splitter.setStretchFactor(1, 3)   

        # 🔹 Fijar un ancho mínimo al panel derecho
        self.tabs.setMinimumWidth(800)  
        self.tabs.setMaximumWidth(800)   

        self.setCentralWidget(splitter)

    # ================================
    # Pestaña "Datos"
    # ================================
    def _tab_datos(self):
        w = QWidget()
        v = QVBoxLayout(w)

        if self.role == "admin":
            btn_rec = QPushButton("📹 Grabar seña (ventana aparte, auto por manos)")
            btn_rec.clicked.connect(self.ui_record_sign)

            btn_rec_clean = QPushButton("🧼 Grabar versión limpia (sin landmarks)")
            btn_rec_clean.clicked.connect(self.ui_record_clean)

            btn_upload = QPushButton("⬆️ Subir seña desde archivo")
            btn_upload.clicked.connect(self.ui_upload_sign)

            btn_edit = QPushButton("✏️ Editar seña")
            btn_edit.clicked.connect(self.ui_edit_sign)

            btn_proc = QPushButton("⚙️ Procesar videos → frames+CSV")
            btn_proc.clicked.connect(self.ui_process_videos)

            btn_train = QPushButton("🧠 Reentrenar ahora")
            btn_train.clicked.connect(self.ui_train)


            # Agregar al layout
            v.addWidget(btn_rec)
            v.addWidget(btn_rec_clean)
            v.addWidget(btn_proc)
            v.addWidget(btn_upload)
            v.addWidget(btn_edit)
            v.addWidget(btn_train)

        v.addStretch(1)
        return w

    def _tab_settings(self):
        w = QWidget()
        v = QVBoxLayout(w)

        self.cb_draw = QCheckBox("Dibujar landmarks (preview)")
        self.cb_draw.setChecked(self.draw_landmarks)

        self.cb_autofill = QCheckBox("Agregar predicciones a la frase (auto)")
        self.cb_autofill.setChecked(self.autofill_sentence)

        row = QHBoxLayout()
        row.addWidget(QLabel("Procesar cada N frames:"))
        sld = QSlider(Qt.Horizontal)
        sld.setRange(1, 10)
        sld.setValue(self.proc_every)
        lbl = QLabel(str(self.proc_every))
        sld.valueChanged.connect(lambda val: lbl.setText(str(val)))
        row.addWidget(sld)
        row.addWidget(lbl)

        apply_btn = QPushButton("Aplicar")
        def _apply():
            self.draw_landmarks = self.cb_draw.isChecked()
            self.proc_every = int(sld.value())
            self.autofill_sentence = self.cb_autofill.isChecked()
        apply_btn.clicked.connect(_apply)

        v.addWidget(self.cb_draw)
        v.addWidget(self.cb_autofill)
        v.addLayout(row)
        v.addWidget(apply_btn)
        v.addStretch(1)
        return w

    
    # ---------- Helpers de background ----------
    def _start_worker(self, do_process: bool, do_train: bool, title="Trabajando…"):
        self._progress = QProgressDialog(title, None, 0, 0, self)
        self._progress.setWindowTitle("Por favor espera")
        self._progress.setCancelButton(None)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.show()

        self._worker = ProcTrainWorker(do_process, do_train, self)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self, success: bool, err: str):
        if self._progress:
            self._progress.close()
            self._progress = None

        # re-indexar labels por si hay nuevas
        self.labels_index = self._build_labels_index()

        if success:
            try:
                self.model = load(SK_MODEL_PATH)
                self.meta = load_meta()
            except Exception as e:
                QMessageBox.warning(self, "Modelo", f"No se pudo cargar el modelo:\n{e}")
                return
            QMessageBox.information(self, "Listo", "Proceso completado correctamente.")
        else:
            QMessageBox.warning(self, "Error", f"Ocurrió un error:\n{err}")

        self._worker = None

    # ---------- Acciones ----------
    def ui_record_sign(self):
        text, ok = QInputDialog.getText(self, "Nueva seña", "Nombre del label (ej. buenos_dias o hola):")
        if not ok:
            return
        label = sanitize_label(text)
        if not label:
            QMessageBox.warning(self, "Etiqueta", "Etiqueta vacía.")
            return

        # Pausar y liberar la cámara principal ANTES de abrir la ventana de grabación
        self._pause_camera(True)
        dlg = RecorderWindow(self, label)
        result = dlg.exec_()

        # Reanudar cámara principal SIEMPRE, haya o no video
        self._pause_camera(False)

        if result == QDialog.Accepted and getattr(dlg, "out_path", None):
            self.last_recorded_video = dlg.out_path
            self.labels_index = self._build_labels_index()

        if result == QDialog.Accepted and AUTO_PROCESS_AFTER_RECORD:
            self._start_worker(do_process=True, do_train=AUTO_TRAIN_AFTER_RECORD,
                               title="Procesando videos y entrenando…")

    def ui_process_videos(self):
        self._start_worker(do_process=True, do_train=False, title="Procesando videos…")

    def ui_train(self):
        self._start_worker(do_process=False, do_train=True, title="Entrenando modelo…")

    # ---------- Loop de cámara principal ----------
    def update_frame(self):
        # Si la cámara no está lista, intenta reabrirla una vez
        if (self.cap is None) or (not self.cap.isOpened()):
            self._open_main_camera()
            if (self.cap is None) or (not self.cap.isOpened()):
                return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Lectura fallida: reintento suave
            self._open_main_camera()
            return

        frame = cv2.flip(frame, 1)

        if self.model and (self.frame_i % self.proc_every == 0):
            small = cv2.resize(frame, PROC_SIZE)
            res = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                if self.draw_landmarks:
                    res_big = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_hands(frame, res_big)
                vec = normalized_twohands_vector(res)
                self._predict_and_update(vec)

        self._blit_to_label(frame)
        self.frame_i += 1

    def _predict_and_update(self, vec):
        try:
            if hasattr(self.model, "predict_proba"):
                Xs = [np.array(vec, dtype=np.float32)]
                proba = self.model.predict_proba(Xs)[0]
                idx = int(np.argmax(proba))
                pred = str(self.model.classes_[idx])
                conf = float(proba[idx])
            else:
                scaler = self.model["scaler"]; clf = self.model["clf"]
                Xs = scaler.transform([np.array(vec, dtype=np.float32)])
                proba = clf.predict_proba(Xs)[0]
                idx = int(np.argmax(proba))
                pred = str(clf.classes_[idx])
                conf = float(proba[idx])

            self.hist.append((pred, conf))

            thr = float(self.meta.get("min_conf", MIN_CONF))
            tokens = [p for (p, c) in self.hist if c >= thr and not p.startswith("__")]
            if not tokens:
                return

            from collections import Counter
            cnt = Counter(tokens)
            top_label, top_votes = cnt.most_common(1)[0]

            required = max(3, int(STABLE_REQUIRE))
            if len(top_label) == 1:
                required = 1

            if top_votes >= required:
                pretty = top_label.replace("_", " ")

                # 👉 SOLO acumula la frase; NO crear burbujas aquí
                if not self.sentence or self.sentence[-1] != pretty:
                    self.sentence.append(pretty)
                    frase_final = self._formatear_frase(" ".join(self.sentence))
                    self.txt_sentence.setPlainText(frase_final)

                # refresco corto para no disparar la misma palabra en loop
                self.hist.clear()

        except Exception as e:
            print("[PRED] Error:", e)



    # ---------- Helpers ----------
    def _blit_to_label(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        pm = pm.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.video_label.setPixmap(pm)

    def _flush_user_sentence_to_chat(self):
        if not self.sentence:
            return None
        frase = self._formatear_frase(" ".join(self.sentence))
        # burbuja a la IZQUIERDA (no oyente)
        self._chat_add_message("user", frase)
        # limpiar buffer visual y lógico
        self.sentence.clear()
        self.txt_sentence.clear()
        return frase

    def _pause_camera(self, pause: bool):
        if pause:
            try:
                if self.timer and self.timer.isActive():
                    self.timer.stop()
            except:
                pass
            try:
                if self.hands:
                    self.hands.close()
                    self.hands = None
            except:
                pass
            try:
                if self.cap:
                    self.cap.release()
                    self.cap = None
            except:
                pass
        else:
            # Reabrir cámara y reiniciar timer
            self._open_main_camera()
            self.frame_i = 0
            try:
                if self.timer and not self.timer.isActive():
                    self.timer.start(33)
            except:
                pass

    def _formatear_frase(self, frase: str) -> str:
        if not frase:
            return ""
        frase = frase.strip().capitalize()

        # Reglas simples de puntuación
        if frase.endswith(("como estas", "como estás")):
            frase += "?"
        elif not frase.endswith((".", "?", "!")):
            frase += "."

        return frase

    # ======= INDEXADO DE LABELS Y VIDEOS =======
    def _build_labels_index(self) -> dict:
        """
        Devuelve un dict:
           { label: {"videos": [rutas...], "mtime": último_mtime} }
        Busca en QUICK_VIDEOS_PATH y FRAME_ACTIONS_PATH/<label>/*.mp4
        """
        index = {}

        # QUICK_VIDEOS_PATH
        if os.path.isdir(QUICK_VIDEOS_PATH):
            for path in glob.glob(os.path.join(QUICK_VIDEOS_PATH, "*.*")):
                if not os.path.isfile(path): continue
                if not re.search(r"\.(mp4|avi|mov|mkv)$", path, re.I): continue
                stem = os.path.splitext(os.path.basename(path))[0]
                m = re.match(r"(.+?)_\d{8}_\d{6}$", stem)
                label = m.group(1) if m else stem
                index.setdefault(label, {"videos": [], "mtime": 0.0})
                index[label]["videos"].append(path)
                index[label]["mtime"] = max(index[label]["mtime"], os.path.getmtime(path))

        # FRAME_ACTIONS_PATH/<label>/*.mp4
        if os.path.isdir(FRAME_ACTIONS_PATH):
            for label in os.listdir(FRAME_ACTIONS_PATH):
                ldir = os.path.join(FRAME_ACTIONS_PATH, label)
                if not os.path.isdir(ldir): continue
                vids = []
                for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                    vids += glob.glob(os.path.join(ldir, ext))
                if vids:
                    index.setdefault(label, {"videos": [], "mtime": 0.0})
                    index[label]["videos"].extend(vids)
                    for p in vids:
                        index[label]["mtime"] = max(index[label]["mtime"], os.path.getmtime(p))

        for label in list(index.keys()):
            index[label]["videos"].sort(key=lambda p: os.path.getmtime(p), reverse=True)

        return index

    def _available_labels(self) -> set:
        return set(self.labels_index.keys())

    def _find_video_for_label(self, label: str):
        info = self.labels_index.get(label)
        if not info: return None
        return info["videos"][0] if info["videos"] else None

    def _phrase_to_labels(self, phrase: str) -> list:
        """
        Convierte una frase libre en una lista de labels existentes.
        Greedy por n-gramas: 3 -> 2 -> 1, usando variantes de label.
        """
        words = _clean_text_to_words(phrase)
        if not words: return []
        labels = []
        avail = self._available_labels()

        i = 0
        while i < len(words):
            matched = None
            for n in (3, 2, 1):
                if i + n > len(words): 
                    continue
                chunk = words[i:i+n]
                for cand in _possible_label_variants(chunk):
                    if cand in avail:
                        matched = cand
                        break
                if matched:
                    break
            if matched:
                labels.append(matched)
                i += n
            else:
                i += 1
        return labels

    def _play_label_sequence(self, labels: list, message_text: str):
        seq = []
        for lb in labels:
            v = self._find_video_for_label(lb)
            if v: seq.append((v, lb))
        if not seq:
            QMessageBox.information(self, "Responder", "No se encontraron videos para ese mensaje.")
            return
        for path, _lb in seq:
            player = PlayerWindow(self, path, model=self.model, meta=self.meta, tokens_text=message_text)
            player.exec_()

    # ======= QUICK REPLIES =======
    def _load_quick_replies(self):
        try:
            if os.path.exists(self.quick_path):
                with open(self.quick_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception as e:
            print("[QR] No se pudo leer quick_replies.json:", e)
        # defaults (mensaje libre)
        return [
            {"title": "Buen día", "message": "buenos dias"},
            {"title": "Solicitar DPI", "message": "buenos dias podria darme su dpi"},
        ]

    def _save_quick_replies(self):
        try:
            os.makedirs(os.path.dirname(self.quick_path), exist_ok=True)
            with open(self.quick_path, "w", encoding="utf-8") as f:
                json.dump(self.quick, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Respuestas", f"No se pudo guardar quick_replies:\n{e}")

    def _tab_responder(self):
        w = QWidget()
        h = QHBoxLayout(w)   

        # --- BLOQUE 1: Chat (izquierda) ---
        chat_frame = QVBoxLayout()
        self.chat_list = QListWidget()
        self.chat_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: #f5f5f5;
                padding: 8px;
            }
            QListWidget::item {
                border-radius: 10px;
                margin: 4px;
                padding: 6px 10px;
                font-size: 14px;
                max-width: 300px;
                word-wrap: true;
            }
        """)
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.chat_list.setWordWrap(True)
        self.chat_list.setUniformItemSizes(False)

        chat_frame.addWidget(self.chat_list)

       # --- INPUT + BOTÓN ENVIAR ---
        row_chat = QHBoxLayout()
        row_chat.setContentsMargins(12, 8, 12, 12)  
        row_chat.setSpacing(10)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Escribe un mensaje...")
        self.chat_input.setMinimumHeight(40)  
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #3498db;
                border-radius: 12px;
                padding: 8px 12px;
                font-size: 14px;
                background-color: #ffffff;
            }
            QLineEdit:focus {
                border: 2px solid #1f6ea1;
            }
        """)

        btn_send = QPushButton("➤")
        btn_send.setObjectName("sendButton")
        btn_send.setFixedSize(36, 36)
        btn_send.setCursor(Qt.PointingHandCursor)
        btn_send.setToolTip("Enviar mensaje")
        btn_send.setStyleSheet("""
            #sendButton {
                background-color: #2980b9;
                color: white;
                font-size: 14px;
                border-radius: 18px;
                border: none;
            }
            #sendButton:hover {
                background-color: #3498db;
            }
            #sendButton:pressed {
                background-color: #2471a3;
            }
        """)

        btn_send.clicked.connect(self._send_chat_message)

        # 🔹 El input ocupa todo el ancho, el botón queda al final
        row_chat.addWidget(self.chat_input, stretch=8)
        row_chat.addWidget(btn_send, stretch=0, alignment=Qt.AlignVCenter)

        chat_frame.addLayout(row_chat)


        chat_widget = QWidget()
        chat_widget.setLayout(chat_frame)
        h.addWidget(chat_widget, 2)

        # --- BLOQUE 2: Respuestas rápidas (derecha) ---
        right = QVBoxLayout()
        self.qr_list = QListWidget()
        self._qr_refresh_list()

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        btn_add = QPushButton("➕ Agregar")
        btn_edit = QPushButton("✏️ Editar")
        btn_delete = QPushButton("🗑️ Borrar")

        btn_add.setObjectName("btnSmall")
        btn_edit.setObjectName("btnSmall")
        btn_delete.setObjectName("btnSmall")

        if self.role == "admin":
            row1.addWidget(btn_add)
            row1.addWidget(btn_edit)
            row1.addWidget(btn_delete)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        btn_use = QPushButton("▶️ Usar / Reproducir")
        btn_play_qr = QPushButton("▶️ Reproducir (solo)")

        btn_use.setObjectName("btnReply")
        btn_play_qr.setObjectName("btnReply")

        row2.addWidget(btn_use)
        row2.addWidget(btn_play_qr)
        row2.addStretch(1)

        if self.role == "admin":
            row3 = QHBoxLayout()
            row3.setSpacing(8)
            btn_rec_qr = QPushButton("🎥 Grabar video")
            btn_attach_qr = QPushButton("📎 Adjuntar video…")
            btn_rec_qr.setObjectName("btnReply")
            btn_attach_qr.setObjectName("btnReply")
            row3.addWidget(btn_rec_qr)
            row3.addWidget(btn_attach_qr)
            row3.addStretch(1)
            right.addLayout(row3)

        right.setSpacing(12)
        right.addWidget(self.qr_list)
        right.addLayout(row1)
        right.addLayout(row2)
        right.addStretch(1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        h.addWidget(right_widget, 1)

        # Conectar botones
        btn_add.clicked.connect(self._qr_add)
        btn_edit.clicked.connect(self._qr_edit)
        btn_delete.clicked.connect(self._qr_del)
        btn_use.clicked.connect(self._qr_use_play)
        btn_play_qr.clicked.connect(self._qr_play_only)
    
        if self.role == "admin":
            btn_rec_qr.clicked.connect(self._qr_record_video_for_selected)
            btn_attach_qr.clicked.connect(self._qr_attach_video_for_selected)

        return w

    

    
    def _chat_add_message(self, sender, text):
        item = QListWidgetItem()

        if sender == "user":
            bg_color = "#d5fdd5"
            align = Qt.AlignLeft
            name = "No oyente"
        else:
            bg_color = "#cfe4ff"
            align = Qt.AlignRight
            name = "Usuario"

        # 🔹 Crea una burbuja visible con el texto en negro
        bubble_html = f"""
        <div style="
            background-color:{bg_color};
            border-radius:12px;
            padding:8px 12px;
            margin:6px;
            color:black;
            font-size:15px;
            word-wrap:break-word;
            white-space:pre-wrap;
            max-width:85%;
        ">
            <b>{name}:</b><br>{text}
        </div>
        """

        label = QLabel()
        label.setText(bubble_html)
        label.setTextFormat(Qt.RichText)
        label.setAlignment(align)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 🔹 Ajusta tamaño al texto
        label.adjustSize()
        item.setSizeHint(label.sizeHint())

        # 🔹 Agregar al chat
        self.chat_list.addItem(item)
        self.chat_list.setItemWidget(item, label)
        self.chat_list.scrollToBottom()


    def _send_chat_message(self):
        msg = self.chat_input.text().strip()
        if not msg:
            return

        # 1) Cierra el turno del no oyente (si hay frase acumulada)
        self._flush_user_sentence_to_chat()

        # 2) Tu respuesta (burbuja a la DERECHA)
        self._chat_add_message("system", msg)
        self.chat_input.clear()

        # 3) (Opcional) reproducir seña de tu respuesta si hay videos
        labels = self._phrase_to_labels(msg)
        if labels:
            self._play_label_sequence(labels, msg)





    def _qr_refresh_list(self):
        self.qr_list.clear()
        for item in self.quick:
            title = item.get("title", "(sin título)")
            message = item.get("message", "")
            tokens = item.get("tokens", []) 
            text = message if message else " ".join(tokens)
            has_vid = " 🎞" if item.get("video") else ""
            li = QListWidgetItem(f"{title}{has_vid}     —     {text}")
            self.qr_list.addItem(li)

    def _qr_current_index(self):
        idx = self.qr_list.currentRow()
        return idx if 0 <= idx < len(self.quick) else -1

    def _qr_add(self):
        title, ok = QInputDialog.getText(self, "Nueva respuesta", "Título:")
        if not ok or not title.strip(): return
        message, ok2 = QInputDialog.getText(
            self, "Mensaje", "Escribe el mensaje (sin guiones bajos):"
        )
        if not ok2: return
        self.quick.append({"title": title.strip(), "message": message.strip()})
        self._save_quick_replies()
        self._qr_refresh_list()

    def _qr_edit(self):
        idx = self._qr_current_index()
        if idx < 0: return
        item = self.quick[idx]
        title0 = item.get("title","")
        message0 = item.get("message", " ".join(item.get("tokens", [])))
        title, ok = QInputDialog.getText(self, "Editar título", "Título:", QLineEdit.Normal, title0)
        if not ok or not title.strip(): return
        message, ok2 = QInputDialog.getText(self, "Editar mensaje", "Mensaje:", QLineEdit.Normal, message0)
        if not ok2: return
        item["title"] = title.strip()
        item["message"] = message.strip()
        self._save_quick_replies()
        self._qr_refresh_list()


        #Panel de administracion 
    def ui_admin_panel(self):
            dlg = QDialog(self)
            dlg.setWindowTitle("Panel de Administración")
            lay = QVBoxLayout(dlg)

            btn_users = QPushButton("👤 Gestionar Usuarios")
            btn_labels = QPushButton("✏️ Gestionar Señas")

            lay.addWidget(btn_users)
            lay.addWidget(btn_labels)

            dlg.exec_()


    def _qr_del(self):
        idx = self._qr_current_index()
        if idx < 0: return
        if QMessageBox.question(self, "Borrar", "¿Eliminar esta respuesta rápida?") != QMessageBox.Yes:
            return
        del self.quick[idx]
        self._save_quick_replies()
        self._qr_refresh_list()

    def _qr_use_play(self):
        idx = self._qr_current_index()
        if idx < 0:
            return
        item = self.quick[idx]
        message = item.get("message", " ".join(item.get("tokens", []))).strip()
        video_path = item.get("video")

        if not message and not video_path:
            QMessageBox.information(self, "Responder", "Esta respuesta no tiene mensaje ni video.")
            return

        # 🔹 Si hay video adjunto, úsalo directamente
        if video_path and os.path.exists(video_path):
            player = PlayerWindow(self, video_path, model=self.model, meta=self.meta, tokens_text=message)
            player.exec_()
            return

        # 🔹 Si no hay video adjunto, intenta reproducir las señas del mensaje
        if message:
            labels = self._phrase_to_labels(message)
            if labels:
                self._play_label_sequence(labels, message)
            else:
                QMessageBox.information(self, "Responder", "No se encontraron señas para este mensaje.")


    def _qr_play_only(self):
        self._qr_use_play()

    def _qr_record_video_for_selected(self):
        idx = self._qr_current_index()
        if idx < 0:
            return
        item = self.quick[idx]
        idx = self._qr_current_index()
        title = item.get("title","respuesta")
        slug = f"{sanitize_label(title)}_{idx}" or f"respuesta_{idx}"

        # 💡 asegurar carpeta y rol
        print(f"[DEBUG] Grabando respuesta rápida en: {QUICK_VIDEOS_PATH}")
        os.makedirs(QUICK_VIDEOS_PATH, exist_ok=True)

        if self.role != "admin":
            QMessageBox.warning(self, "Permiso denegado", "Solo los administradores pueden grabar videos.")
            return

        self._pause_camera(True)
        try:
            dlg = RecorderWindow(self, label="quickreply", dest_folder=QUICK_VIDEOS_PATH)
            result = dlg.exec_()
        finally:
            self._pause_camera(False)

        if result == QDialog.Accepted and getattr(dlg, "out_path", None):
            item["video"] = dlg.out_path
            self.last_recorded_video = dlg.out_path
            self._save_quick_replies()
            self._qr_refresh_list()
            self.labels_index = self._build_labels_index()
            QMessageBox.information(self, "Respuestas", "Video guardado y vinculado.")


    def _qr_attach_video_for_selected(self):
        idx = self._qr_current_index()
        if idx < 0: return
        fn, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", QUICK_VIDEOS_PATH,
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if not fn: 
            return
        self.quick[idx]["video"] = fn
        self._save_quick_replies()
        self._qr_refresh_list()
        self.labels_index = self._build_labels_index()
        QMessageBox.information(self, "Respuestas", "Video adjuntado a la respuesta.")

    # ======= COMPOSITOR =======
    def _qr_compose_attach_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", QUICK_VIDEOS_PATH,
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if not fn:
            return
        self.compose_video_path = fn
        base = os.path.basename(fn)
        self.lbl_compose_vid.setText(f"Video adjunto: {base}")

    def _qr_compose_record_video(self):
        title = self.compose_title.text().strip() or "custom"
        slug  = sanitize_label(title)
        self._pause_camera(True)
        try:
            dlg = RecorderWindow(self, label=slug, dest_folder=QUICK_VIDEOS_PATH)
            result = dlg.exec_()
        finally:
            self._pause_camera(False)

    def _qr_compose_use_text(self):
        message = self.compose_message.text().strip()
        if not message:
            QMessageBox.information(self, "Compositor", "Escribe un mensaje.")
            return

        if self.compose_video_path and os.path.exists(self.compose_video_path):
            player = PlayerWindow(self, self.compose_video_path, model=self.model, meta=self.meta, tokens_text=message)
            player.exec_()
            return

        labels = self._phrase_to_labels(message)
        self._play_label_sequence(labels, message)
    def ui_add_detected(self):
        """Acción para el botón ➕ Añadir detectada"""
        if self.detected_word and self.detected_word != "...":
            pretty = self.detected_word.replace("_", " ")
            QMessageBox.information(self, "Añadir detectada",
                                    f"Se agregó la seña detectada: {pretty}")
            print(f"[ADD] Seña agregada: {pretty}")
        else:
            QMessageBox.warning(self, "Añadir detectada",
                                "No hay ninguna seña detectada.")

    def _qr_compose_save(self):
        title = self.compose_title.text().strip()
        message = self.compose_message.text().strip()
        if not title:
            QMessageBox.information(self, "Compositor", "Escribe un título.")
            return
        if not message:
            QMessageBox.information(self, "Compositor", "Escribe un mensaje.")
            return

        item = {"title": title, "message": message}
        if self.compose_video_path and os.path.exists(self.compose_video_path):
            item["video"] = self.compose_video_path

        self.quick.append(item)
        self._save_quick_replies()
        self._qr_refresh_list()

        self.compose_title.clear()
        self.compose_message.clear()
        self.compose_video_path = None
        self.lbl_compose_vid.setText("Video adjunto: (ninguno)")
        QMessageBox.information(self, "Respuestas", "Respuesta rápida guardada.")

        
    # respuesta rápida
    def _qr_compose_use_text(self):
        message = self.compose_message.text().strip()
        if not message:
            QMessageBox.information(self, "Compositor", "Escribe un mensaje.")
            return

        # Guardar en historial
        ts = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "history_list"):
            self.history_list.addItem(f"[{ts}] RESPUESTA: {message}")

        # Buscar videos
        labels = self._phrase_to_labels(message)
        self._play_label_sequence(labels, message)

        # Limpiar entrada
        self.compose_message.clear()

    def closeEvent(self, e):
        # Cerrar limpio todo
        try:
            if self.timer and self.timer.isActive():
                self.timer.stop()
        except:
            pass
        try: 
            if self.hands:
                self.hands.close()
                self.hands = None
        except: 
            pass
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except: 
            pass
        cv2.destroyAllWindows()
        e.accept()
        
    def _generate_thumbnail(self, video_path, thumb_path):
            cap = cv2.VideoCapture(video_path)
            ok, frame = cap.read()
            cap.release()
            if ok:
                frame = cv2.resize(frame, (160, 120))
                cv2.imwrite(thumb_path, frame)


    def _build_labels_index(self):
            index = {}
            for label in os.listdir(FRAME_ACTIONS_PATH):
                ldir = os.path.join(FRAME_ACTIONS_PATH, label)
                if not os.path.isdir(ldir): continue
                vids = glob.glob(os.path.join(ldir, "*.mp4"))
                if vids:
                    thumb_path = os.path.join(ldir, "thumb.jpg")
                    if not os.path.exists(thumb_path):
                        self._generate_thumbnail(vids[0], thumb_path)
                    index[label] = {"videos": vids, "thumb": thumb_path}
            return index
    #historial 
    def _tab_history(self):
        w = QWidget()
        v = QVBoxLayout(w)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._history_preview)
        v.addWidget(self.history_list)
        return w
    
    #vista previa historial 
    def _history_preview(self, item):
        """
        Vista previa cuando haces doble clic en un historial detectado.
        """
        text = item.text()
        # Extrae la palabra (quita timestamp)
        if "] " in text:
            pretty = text.split("] ", 1)[1]
        else:
            pretty = text

        # Busca si existe un video para esa palabra
        label = pretty.replace(" ", "_")  # normalizamos
        video_path = self._find_video_for_label(label)

        if video_path:
            player = PlayerWindow(self, video_path, model=None, meta=None, tokens_text=pretty)
            player.exec_()
        else:
            QMessageBox.information(self, "Vista previa", f"No hay video para '{pretty}'.")

    #Edición de seña
    def ui_edit_sign(self):
        labels = list(self._available_labels())
        if not labels:
            QMessageBox.information(self, "Editar", "No hay señas guardadas.")
            return

        label, ok = QInputDialog.getItem(self, "Editar seña", "Selecciona la seña:", labels, 0, False)
        if not ok or not label:
            return

        videos = self.labels_index.get(label, {}).get("videos", [])
        if not videos:
            QMessageBox.information(self, "Editar", f"No hay videos para la seña '{label}'.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Editar seña: {label}")
        layout = QVBoxLayout(dlg)

        list_widget = QListWidget()
        list_widget.setViewMode(QListWidget.IconMode)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(100, 400)
        slider.setValue(200)
        slider.valueChanged.connect(lambda val: list_widget.setIconSize(QSize(val, val // 2)))
        layout.addWidget(slider)
        list_widget.setResizeMode(QListWidget.Adjust)
        list_widget.setMovement(QListWidget.Static)
        list_widget.setSpacing(10)

        for v in videos:
            item = QListWidgetItem(os.path.basename(v))
            item.setData(Qt.UserRole, v)
            thumb_path = os.path.splitext(v)[0] + "_thumb.jpg"
            if not os.path.exists(thumb_path):
                self._generate_thumbnail(v, thumb_path)
            if os.path.exists(thumb_path):
                icon = QIcon(thumb_path)
                item.setIcon(icon)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        # Botones
        btn_preview = QPushButton("▶️ Reproducir")
        btn_delete = QPushButton("🗑️ Eliminar")
        btn_replace = QPushButton("🔄 Reemplazar")
        btn_add = QPushButton("➕ Agregar nuevo")
        btn_close = QPushButton("Cerrar")

        row = QHBoxLayout()
        for b in (btn_preview, btn_delete, btn_replace, btn_add, btn_close):
            row.addWidget(b)
        layout.addLayout(row)

            # --- Funciones ---
        def do_preview():
            item = list_widget.currentItem()
            if not item:
                return
            video_path = item.data(Qt.UserRole)
            player = PlayerWindow(self, video_path, model=None, meta=None, tokens_text=f"{label}")
            player.exec_()

        def do_delete():
            item = list_widget.currentItem()
            if not item:
                return
            video_path = item.data(Qt.UserRole)
            label_folder = os.path.dirname(video_path)

            resp = QMessageBox.question(
                self, "Eliminar",
                f"¿Deseas eliminar solo este video o toda la seña completa '{os.path.basename(label_folder)}'?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            if resp == QMessageBox.Cancel:
                return

            import shutil
            try:
                if resp == QMessageBox.Yes:
                    os.remove(video_path)
                    list_widget.takeItem(list_widget.row(item))
                elif resp == QMessageBox.No:
                    shutil.rmtree(label_folder)
                    QMessageBox.information(self, "Eliminado", f"Se eliminó la seña completa '{os.path.basename(label_folder)}'.")
                    dlg.accept()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo eliminar:\n{e}")

        def do_replace():
            item = list_widget.currentItem()
            if not item:
                return
            old_path = item.data(Qt.UserRole)
            fn, _ = QFileDialog.getOpenFileName(dlg, "Selecciona nuevo video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
            if fn:
                import shutil
                shutil.copy(fn, old_path)
                QMessageBox.information(dlg, "Reemplazo", f"{os.path.basename(old_path)} fue reemplazado.")

        def do_add():
            fn, _ = QFileDialog.getOpenFileName(dlg, "Selecciona video a agregar", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
            if fn:
                import shutil
                folder = os.path.join(FRAME_ACTIONS_PATH, label)
                ensure_dir(folder)
                base = os.path.basename(fn)
                dest = os.path.join(folder, base)
                shutil.copy(fn, dest)
                item = QListWidgetItem(base)
                item.setData(Qt.UserRole, dest)
                list_widget.addItem(item)
                QMessageBox.information(dlg, "Agregar", f"Nuevo video agregado a {label}")

        def do_close():
            dlg.accept()

        # 🔹 Conectar los botones
        btn_preview.clicked.connect(do_preview)
        btn_delete.clicked.connect(do_delete)
        btn_replace.clicked.connect(do_replace)
        btn_add.clicked.connect(do_add)
        btn_close.clicked.connect(do_close)

        dlg.exec_()

        # Reindexar y preguntar si quiere reentrenar
        self.labels_index = self._build_labels_index()
        if QMessageBox.question(self, "Reentrenar", "¿Quieres reentrenar el modelo ahora?") == QMessageBox.Yes:
            self._start_worker(do_process=True, do_train=True, title="Procesando y entrenando…")

class AdminMenu(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Panel de Administración")
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        btn_atender = QPushButton("👂 Atender no oyentes ")
        btn_atender.clicked.connect(self._as_user)

        btn_senas = QPushButton("✋ Gestionar señas")
        btn_senas.clicked.connect(self._manage_signs)

        btn_users = QPushButton("👥 Gestionar usuarios")
        btn_users.clicked.connect(self._manage_users)

  
        for b in (btn_atender, btn_senas, btn_users):
            layout.addWidget(b)

        self.choice = None  

    def _as_user(self):
        self.choice = "user_mode"  
        self.accept()

    def _manage_signs(self):
        self.choice = "signs"
        self.accept()

    def _manage_users(self):
        self.choice = "users"
        self.accept()

    def _view_logs(self):
        self.choice = "logs"
        self.accept()
