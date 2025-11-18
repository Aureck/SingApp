import os
import sys
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QIcon   
from mainwindow import LSPDashboard  
from helpers import get_data_path  

def load_styles(app, qss_path="styles/main.qss"):
    try:
        style_path = get_data_path(qss_path)
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            print(f"[STYLE] Estilo cargado desde: {style_path}")
    except Exception as e:
        print(f"[WARN] No se pudo aplicar estilo ({qss_path}): {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ============  AUTO-ACTUALIZADOR  ============
    try:
        from updater import check_for_update
        check_for_update()
    except Exception as e:
        print("[UPDATE] Error al verificar actualización:", e)
    # ================================================

    icon_path = get_data_path("icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        print(f"[ICON] Icono cargado desde: {icon_path}")
    else:
        print("[ICON] No se encontró icon.ico")

    load_styles(app, "styles/main.qss")

    from login_window import LoginWindow
    login = LoginWindow()

    if os.path.exists(icon_path):
        login.setWindowIcon(QIcon(icon_path))

    if login.exec_() == QDialog.Accepted:

        if login.role == "admin":
            from admin_menu import AdminMenu

            while True:
                menu = AdminMenu()
                if os.path.exists(icon_path):
                    menu.setWindowIcon(QIcon(icon_path))  

                if menu.exec_() != QDialog.Accepted:
                    break  # salir del ciclo si cierra el panel

                # --- Opción 1: modo atención ---
                if menu.choice == "user_mode":
                    w = LSPDashboard(role="admin", section_title="Atención a usuarios no oyentes")
                    w.setWindowIcon(QIcon(icon_path))  
                    w.showMaximized()
                    app.exec_()

                # --- Opción 2: gestión de señas ---
                elif menu.choice == "signs":
                    w = LSPDashboard(role="admin", section_title="Gestión de señas y videos")
                    w.tabs.setCurrentWidget(w.tab_datos)
                    w.setWindowIcon(QIcon(icon_path)) 
                    w.showMaximized()
                    app.exec_()

                # --- Opción 3: gestión de usuarios ---
                elif menu.choice == "users":
                    from users_window import UsersWindow
                    dlg = UsersWindow()
                    dlg.setWindowTitle("Administración de usuarios del sistema")
                    dlg.setWindowIcon(QIcon(icon_path))  
                    dlg.exec_()

        else:
            w = LSPDashboard(role="usuario", section_title="Modo usuario")
            w.setWindowIcon(QIcon(icon_path))  
            w.showMaximized()
            sys.exit(app.exec_())
