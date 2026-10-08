import sys
from PyQt5.QtWidgets import QApplication
from app.gui import VehicleGUI


def main():
    app = QApplication(sys.argv)
    window = VehicleGUI()
    window.show()
    sys.exit(
        app.exec_()
    )
if __name__ == "__main__":
    main()