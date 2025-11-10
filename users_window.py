from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QLineEdit, QLabel, QMessageBox, QComboBox, QWidget, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class UsersWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Administración de usuarios del sistema")
        self.resize(650, 450)
        self.setStyleSheet("""
            QDialog {
                background-color: #f4f8fb;
            }
            QListWidget {
                border: 2px solid #d0dbe8;
                border-radius: 10px;
                font-size: 16px;
                padding: 8px;
                background: white;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                font-size: 17px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 20px;
                border: 2px solid #1f5a85;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1f6ea6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 🔹 Título principal
        title = QLabel("👥 Gestión de usuarios del sistema")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        layout.addWidget(title)

        # 🔹 Lista de usuarios
        self.list_users = QListWidget()
        layout.addWidget(self.list_users)

        # 🔹 Botones CRUD
        row = QHBoxLayout()
        row.setSpacing(15)

        btn_add = QPushButton("➕ Agregar usuario")
        btn_edit = QPushButton("✏️ Editar usuario")
        btn_delete = QPushButton("🗑️ Eliminar usuario")

        row.addWidget(btn_add)
        row.addWidget(btn_edit)
        row.addWidget(btn_delete)
        layout.addLayout(row)

        # Datos en memoria
        self.users = [
            {"username": "admin", "role": "admin", "password": "1234"},
            {"username": "usuario", "role": "usuario", "password": "abcd"},
        ]
        self._refresh_list()

        # Conexiones
        btn_add.clicked.connect(self.add_user)
        btn_edit.clicked.connect(self.edit_user)
        btn_delete.clicked.connect(self.delete_user)

    # =====================
    # FUNCIONES CRUD
    # =====================

    def _refresh_list(self):
        self.list_users.clear()
        for u in self.users:
            self.list_users.addItem(f"👤 {u['username']}   •   Rol: {u['role']}")

    def add_user(self):
        dialog = UserEditDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            self.users.append(data)
            self._refresh_list()

    def edit_user(self):
        idx = self.list_users.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "Editar usuario", "Selecciona un usuario para editar.")
            return
        user = self.users[idx]
        dialog = UserEditDialog(self, user)
        if dialog.exec_() == QDialog.Accepted:
            self.users[idx] = dialog.get_data()
            self._refresh_list()

    def delete_user(self):
        idx = self.list_users.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "Eliminar usuario", "Selecciona un usuario para eliminar.")
            return

        user = self.users[idx]
        confirm = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Eliminar usuario '{user['username']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.users.pop(idx)
            self._refresh_list()


# =====================
# DIÁLOGO DE EDICIÓN / CREACIÓN
# =====================

class UserEditDialog(QDialog):
    def __init__(self, parent=None, user=None):
        super().__init__(parent)
        self.setWindowTitle("Editar usuario" if user else "Nuevo usuario")
        self.resize(400, 250)
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
            QLabel {
                font-size: 15px;
                color: #2c3e50;
            }
            QLineEdit, QComboBox {
                font-size: 15px;
                padding: 6px;
                border: 2px solid #cfd8e3;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                font-size: 15px;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.txt_username = QLineEdit()
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.Password)
        self.cmb_role = QComboBox()
        self.cmb_role.addItems(["usuario", "admin"])

        form.addRow("Nombre de usuario:", self.txt_username)
        form.addRow("Contraseña:", self.txt_password)
        form.addRow("Rol:", self.cmb_role)
        layout.addLayout(form)

        # Botones
        row_btn = QHBoxLayout()
        btn_ok = QPushButton("💾 Guardar")
        btn_cancel = QPushButton("Cancelar")
        row_btn.addWidget(btn_ok)
        row_btn.addWidget(btn_cancel)
        layout.addLayout(row_btn)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        if user:
            self.txt_username.setText(user["username"])
            self.txt_password.setText(user["password"])
            self.cmb_role.setCurrentText(user["role"])

    def get_data(self):
        return {
            "username": self.txt_username.text().strip(),
            "password": self.txt_password.text().strip(),
            "role": self.cmb_role.currentText(),
        }
