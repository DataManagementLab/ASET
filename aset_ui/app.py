"""Main entry-point into the PyQt application."""

import logging

from PyQt5.QtWidgets import QApplication

from aset_ui.mainwindow import MainWindow

logger = logging.getLogger(__name__)


def run(args: [str]):
    """
    Set up and run the application.

    :param args: commandline arguments to pass to the PyQt application
    :return: exit code
    """

    # set up PyQt application
    app = QApplication(args)
    window = MainWindow()
    window.show()
    logger.debug("Initialized the application.")

    return app.exec_()
