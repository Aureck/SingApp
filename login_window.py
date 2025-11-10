from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inicio de Sesión")
        self.resize(600, 350)

        self.role = None  

        # -------- Panel izquierdo con logo --------
        self.logo = QLabel()
        from helpers import get_data_path
        pix = QPixmap(get_data_path("logo_muni.jpg"))

        self.logo.setPixmap(pix.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignCenter)

        left_layout = QVBoxLayout()
        left_layout.addStretch(1)
        left_layout.addWidget(self.logo)
        left_layout.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setStyleSheet("""
            background-color: qlineargradient(
                spread:pad, x1:0, y1:0, x2:1, y2:1,
                stop:0 #1e5799, stop:1 #2989d8
            );
        """)

        # -------- Panel derecho con formulario --------
        title = QLabel("Inicio de sesión")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.txt_user = QLineEdit()
        self.txt_user.setPlaceholderText("Usuario")
        self.txt_pass = QLineEdit()
        self.txt_pass.setPlaceholderText("Contraseña")
        self.txt_pass.setEchoMode(QLineEdit.Password)

        self.btn_login = QPushButton("Iniciar sesión")

        # -------- Estilos --------
        style = """
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #2a7de1;
            }
            QPushButton {
                background-color: #2a7de1;
                color: white;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1d5ba7;
            }
        """
        self.setStyleSheet(style)

        # Layout derecho
        right_layout = QVBoxLayout()
        right_layout.addWidget(title)
        right_layout.addSpacing(20)
        right_layout.addWidget(self.txt_user)
        right_layout.addWidget(self.txt_pass)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.btn_login)
        right_layout.addStretch(1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # -------- Layout principal --------
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

        # -------- Conexiones --------
        self.btn_login.clicked.connect(self._login_user)

    def _login_user(self):
        user = self.txt_user.text().strip()
        password = self.txt_pass.text().strip()
        # Aquí validarías contra tu BD
        if user == "admin" and password == "1234":
            self.role = "admin"
            self.accept()
        elif user == "usuario" and password == "1234":
            self.role = "usuario"
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Usuario o contraseña incorrectos.")
