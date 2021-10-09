import logging
import sys

from PyQt6.QtWidgets import QApplication, QStyleFactory

from aset.resources import ResourceManager
from aset_ui.main_window import MainWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Starting aset_ui.")

    with ResourceManager() as resource_manager:
        # set up PyQt application
        app = QApplication(sys.argv)

        preferred_style = "Windows"
        if preferred_style in QStyleFactory.keys():
            app.setStyle(preferred_style)
            logger.info(f"Using preferred style '{preferred_style}'.")
        else:
            logger.info(f"Using default style '{app.style().objectName()}'.")

        window = MainWindow()

        sys.exit(app.exec())
