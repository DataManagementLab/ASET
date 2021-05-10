"""Main window of the application."""
import logging

from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFormLayout

from aset.core.resources import close_all_resources
from aset_ui.offlinephase import OfflinePhaseWindow
from aset_ui.onlinephase import OnlinePhaseWindow

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window of the application."""

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("ASET")

        # set up the widgets
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()

        self.header = QLabel("ASET: Ad-hoc Structured Exploration of Text Collections")
        header_font = self.header.font()
        header_font.setPointSize(20)
        header_font.setWeight(100)
        self.header.setFont(header_font)
        self.central_layout.addWidget(self.header)

        self.buttons_form_widget = QWidget()
        self.buttons_form_layout = QFormLayout()
        self.buttons_form_layout.setContentsMargins(0, 0, 0, 0)

        self.offline_phase_label = QLabel("Choose a folder containing a document collection to preprocess:")
        self.offline_phase_button = QPushButton("Offline Preprocessing Phase")
        self.offline_phase_button.clicked.connect(self.start_offline_phase)
        self.buttons_form_layout.addRow(self.offline_phase_label, self.offline_phase_button)

        self.online_phase_label = QLabel("Choose a preprocessed document collection and match it to a query:")
        self.online_phase_button = QPushButton("Online Matching Phase")
        self.online_phase_button.clicked.connect(self.start_online_phase)
        self.buttons_form_layout.addRow(self.online_phase_label, self.online_phase_button)

        self.buttons_form_widget.setLayout(self.buttons_form_layout)
        self.central_layout.addWidget(self.buttons_form_widget)

        self.central_widget.setLayout(self.central_layout)
        self.setCentralWidget(self.central_widget)

        # lock the window size
        self.setFixedSize(self.sizeHint())

        # prepare the other windows
        self.offline_phase_window = None
        self.online_phase_window = None

        logger.debug("Initialized main window.")

    def start_offline_phase(self, _):
        """Start the offline phase."""
        logger.info("Start the offline phase.")
        self.offline_phase_window = OfflinePhaseWindow(self)
        self.hide()
        self.offline_phase_window.show()

    def start_online_phase(self, _):
        """Start the online phase."""
        logger.info("Start the online phase.")
        self.online_phase_window = OnlinePhaseWindow(self)
        self.hide()
        self.online_phase_window.show()

    def closeEvent(self, _):
        logger.info("Close application.")
        close_all_resources()
