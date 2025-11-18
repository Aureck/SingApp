from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class AdminMenu(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(" Panel de Administración")
        self.resize(900, 600)  
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f7fa;
            }
            QLabel {
                font-size: 26px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 25px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 12px;
                padding: 14px 20px;
                margin: 8px 0;
                border: 2px solid #1f618d;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
            QPushButton:pressed, QPushButton[active="true"] {
                background-color: #2e86c1;
                border: 2px solid #154360;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # --- Título principal ---
        title = QLabel("⚙️ Panel de Administración")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Imagen debajo del título ---
        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        img.setPixmap(QPixmap("styles/cohe.png").scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(img)

        # --- Botones grandes ---
        self.btn_user_mode = QPushButton("👂 Atención a usuarios no oyentes")
        self.btn_signs     = QPushButton("✋ Gestionar señas y videos")
        self.btn_users     = QPushButton("👥 Administrar usuarios del sistema")

        for b in (self.btn_user_mode, self.btn_signs, self.btn_users):
            b.setMinimumWidth(350)
            layout.addWidget(b)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.choice = None

        self.btn_user_mode.clicked.connect(lambda: self._select("user_mode", self.btn_user_mode))
        self.btn_signs.clicked.connect(lambda: self._select("signs", self.btn_signs))
        self.btn_users.clicked.connect(lambda: self._select("users", self.btn_users))

    def _select(self, choice, btn):
        for b in [self.btn_user_mode, self.btn_signs, self.btn_users]:
            b.setProperty("active", False)
            b.style().unpolish(b)
            b.style().polish(b)

        btn.setProperty("active", True)
        btn.style().unpolish(btn)
        btn.style().polish(btn)

        self.choice = choice
        self.accept()
