from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit

class LogsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Logs del sistema")
        self.resize(500, 400)

        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setText("Aquí se mostrarían los logs...\nEjemplo:\n[10:15] Usuario admin inició sesión\n[10:20] Se agregó seña 'hola'")
        layout.addWidget(self.text)
