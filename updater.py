import requests, os, tempfile, shutil, sys
from PyQt5.QtWidgets import QMessageBox

UPDATE_URL = "https://raw.githubusercontent.com/Aureck/SingApp/main/update.json"
DOWNLOAD_URL = "https://github.com/Aureck/SingApp/releases/latest/download/SingApp.exe"

def check_for_update():
    # Leer versión actual
    try:
        import version
        current = version.APP_VERSION
    except:
        current = "0.0.0"

    # Cargar update.json
    try:
        r = requests.get(UPDATE_URL, timeout=5)
        data = r.json()
    except Exception as e:
        print("No se pudo obtener update.json:", e)
        return

    latest = data.get("latest", current)
    notes = data.get("notes", "")

    # Comparar
    if latest != current:
        ask_update(latest, notes)


def ask_update(latest, notes):
    msg = QMessageBox()
    msg.setWindowTitle("Actualización disponible")
    msg.setText(f"Está disponible la versión {latest}.\n\nNotas:\n{notes}\n\n¿Actualizar ahora?")
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

    if msg.exec_() == QMessageBox.Yes:
        download_update()


def download_update():
    try:
        tmp = tempfile.mkdtemp()
        new_exe = os.path.join(tmp, "SingApp_new.exe")

        r = requests.get(DOWNLOAD_URL, stream=True)
        with open(new_exe, "wb") as f:
            for chunk in r.iter_content(1024 * 100):
                if chunk:
                    f.write(chunk)

        replace_exe(new_exe)

    except Exception as e:
        QMessageBox.warning(None, "Error", f"No se pudo descargar la actualización:\n{e}")


def replace_exe(new_exe):
    current_exe = sys.argv[0]

    try:
        backup = current_exe + ".old"
        if os.path.exists(backup):
            os.remove(backup)

        os.rename(current_exe, backup)
        shutil.copy(new_exe, current_exe)

        QMessageBox.information(None, "Actualización",
                                "Actualización instalada. La app se reiniciará.")
        os.execv(current_exe, sys.argv)

    except Exception as e:
        QMessageBox.warning(None, "Error", f"No se pudo reemplazar el ejecutable:\n{e}")
