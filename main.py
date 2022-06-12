import logging
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from aset.resources import ResourceManager
from aset_ui.main_window import MainWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Starting aset_ui.")

    with ResourceManager() as resource_manager:
        # set up PyQt application
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon("aset_ui/resources/logo.png"))

        window = MainWindow()

        sys.exit(app.exec())
