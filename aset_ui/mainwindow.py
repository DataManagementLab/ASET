"""Main window of the application."""
import logging

from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout

from aset.core.resources import close_all_resources
from aset_ui.offlinephase import OfflinePhaseWindow
from aset_ui.onlinephase import OnlinePhaseWindow
from aset_ui.util import HEADER_FONT, SUBHEADER_FONT, LABEL_FONT

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
        self.header.setFont(HEADER_FONT)
        self.central_layout.addWidget(self.header)

        self.columns_widget = QWidget()
        self.columns_layout = QHBoxLayout()

        # left column
        self.left_column_widget = QWidget()
        self.left_column_layout = QVBoxLayout()

        self.left_subheader = QLabel("1. Offline Preprocessing Phase")
        self.left_subheader.setFont(SUBHEADER_FONT)
        self.left_column_layout.addWidget(self.left_subheader)

        self.left_label = QLabel("Choose a directory containing a document collection and preprocess it.")
        self.left_label.setFont(LABEL_FONT)
        self.left_label.setWordWrap(True)
        self.left_column_layout.addWidget(self.left_label)

        self.left_button = QPushButton("Start Preprocessing Phase")
        self.left_button.clicked.connect(self.start_offline_phase)
        self.left_column_layout.addWidget(self.left_button)

        self.left_column_widget.setLayout(self.left_column_layout)
        self.columns_layout.addWidget(self.left_column_widget)

        # right column
        self.right_column_widget = QWidget()
        self.right_column_layout = QVBoxLayout()

        self.right_subheader = QLabel("2. Online Matching Phase")
        self.right_subheader.setFont(SUBHEADER_FONT)
        self.right_column_layout.addWidget(self.right_subheader)

        self.right_label = QLabel("Open a preprocessed document collection and match it to some attributes.")
        self.right_label.setFont(LABEL_FONT)
        self.right_label.setWordWrap(True)
        self.right_column_layout.addWidget(self.right_label)

        self.right_button = QPushButton("Start Matching Phase")
        self.right_button.clicked.connect(self.start_online_phase)
        self.right_column_layout.addWidget(self.right_button)

        self.right_column_widget.setLayout(self.right_column_layout)
        self.columns_layout.addWidget(self.right_column_widget)

        self.columns_widget.setLayout(self.columns_layout)
        self.central_layout.addWidget(self.columns_widget)

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
