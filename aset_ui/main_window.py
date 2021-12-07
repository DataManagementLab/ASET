import logging

from PyQt6.QtCore import QMutex, Qt, QThread, QWaitCondition, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, \
    QVBoxLayout, QWidget

from aset.data.data import ASETDocumentBase
from aset.matching.phase import BaseMatchingPhase
from aset.preprocessing.phase import PreprocessingPhase
from aset.statistics import Statistics
from aset_ui.aset_api import ASETAPI
from aset_ui.document_base import CreateDocumentBaseWidget, DocumentBaseViewerWidget
from aset_ui.interactive_matching import InteractiveMatchingWidget
from aset_ui.style import BUTTON_FONT, HEADER_FONT, MENU_FONT, STATUS_BAR_FONT

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    ################################
    # signals (aset ui --> aset api)
    ################################
    create_document_base = pyqtSignal(str, list)
    load_document_base_from_bson = pyqtSignal(str)
    save_document_base_to_bson = pyqtSignal(str, ASETDocumentBase)
    save_table_to_csv = pyqtSignal(str, ASETDocumentBase)
    load_default_preprocessing_phase = pyqtSignal()
    load_preprocessing_phase_from_config = pyqtSignal(str)
    save_preprocessing_phase_to_config = pyqtSignal(str, PreprocessingPhase)
    load_default_matching_phase = pyqtSignal()
    load_matching_phase_from_config = pyqtSignal(str)
    save_matching_phase_to_config = pyqtSignal(str, BaseMatchingPhase)
    run_preprocessing_phase = pyqtSignal(ASETDocumentBase, PreprocessingPhase, Statistics)
    run_matching_phase = pyqtSignal(ASETDocumentBase, BaseMatchingPhase, Statistics)
    save_statistics_to_json = pyqtSignal(str, Statistics)

    ##############################
    # slots (aset api --> aset ui)
    ##############################
    @pyqtSlot(str, float)
    def status(self, message, progress):
        logger.debug("Called slot 'status'.")
        self.status_widget_message.setText(message)
        if progress == -1:
            self.status_widget_progress.setRange(0, 0)
        else:
            self.status_widget_progress.setRange(0, 100)
            self.status_widget_progress.setValue(int(progress * 100))

    @pyqtSlot(str)
    def finished(self, message):
        logger.debug("Called slot 'finished'.")
        self.status_widget_message.setText(message)
        self.status_widget_progress.setRange(0, 100)
        self.status_widget_progress.setValue(100)

        self.show_document_base_viewer_widget()
        self._enable_global_input()

    @pyqtSlot(str)
    def error(self, message):
        logger.debug("Called slot 'error'.")
        self.status_widget_message.setText(message)
        self.status_widget_progress.setRange(0, 100)
        self.status_widget_progress.setValue(100)

        self.show_document_base_viewer_widget()
        self._enable_global_input()

    @pyqtSlot(ASETDocumentBase)
    def document_base_to_ui(self, document_base):
        logger.debug("Called slot 'document_base_to_ui'.")
        self.document_base = document_base

        self.document_base_viewer_widget.update_document_base(self.document_base)

        self.save_document_base_to_bson_action.setEnabled(True)
        self.save_table_to_csv_action.setEnabled(True)
        if self.preprocessing_phase is not None:
            self.run_preprocessing_phase_action.setEnabled(True)
        if self.matching_phase is not None:
            self.run_matching_phase_action.setEnabled(True)

    @pyqtSlot(PreprocessingPhase)
    def preprocessing_phase_to_ui(self, preprocessing_phase):
        logger.debug("Called slot 'preprocessing_phase_to_ui'.")
        self.preprocessing_phase = preprocessing_phase

        self.save_preprocessing_phase_to_config_action.setEnabled(True)
        if self.document_base is not None:
            self.run_preprocessing_phase_action.setEnabled(True)

    @pyqtSlot(BaseMatchingPhase)
    def matching_phase_to_ui(self, matching_phase):
        logger.debug("Called slot 'matching_phase_to_ui'.")
        self.matching_phase = matching_phase

        self.save_matching_phase_to_config_action.setEnabled(True)
        if self.document_base is not None:
            self.run_matching_phase_action.setEnabled(True)

    @pyqtSlot(Statistics)
    def statistics_to_ui(self, statistics):
        logger.debug("Called slot 'statistics_to_ui'.")
        self.statistics = statistics

    @pyqtSlot(dict)
    def feedback_request_to_ui(self, feedback_request):
        logger.debug("Called slot 'feedback_request_to_ui'.")
        self.interactive_matching_widget.handle_feedback_request(feedback_request)

    # noinspection PyUnresolvedReferences
    def _connect_slots_and_signals(self):
        self.create_document_base.connect(self.api.create_document_base)
        self.load_document_base_from_bson.connect(self.api.load_document_base_from_bson)
        self.save_document_base_to_bson.connect(self.api.save_document_base_to_bson)
        self.save_table_to_csv.connect(self.api.save_table_to_csv)
        self.load_default_preprocessing_phase.connect(self.api.load_default_preprocessing_phase)
        self.load_preprocessing_phase_from_config.connect(self.api.load_preprocessing_phase_from_config)
        self.save_preprocessing_phase_to_config.connect(self.api.save_preprocessing_phase_to_config)
        self.load_default_matching_phase.connect(self.api.load_default_matching_phase)
        self.load_matching_phase_from_config.connect(self.api.load_matching_phase_from_config)
        self.save_matching_phase_to_config.connect(self.api.save_matching_phase_to_config)
        self.run_preprocessing_phase.connect(self.api.run_preprocessing_phase)
        self.run_matching_phase.connect(self.api.run_matching_phase)
        self.save_statistics_to_json.connect(self.api.save_statistics_to_json)

        self.api.status.connect(self.status)
        self.api.finished.connect(self.finished)
        self.api.error.connect(self.error)
        self.api.document_base_to_ui.connect(self.document_base_to_ui)
        self.api.preprocessing_phase_to_ui.connect(self.preprocessing_phase_to_ui)
        self.api.matching_phase_to_ui.connect(self.matching_phase_to_ui)
        self.api.statistics_to_ui.connect(self.statistics_to_ui)
        self.api.feedback_request_to_ui.connect(self.feedback_request_to_ui)

    ###################
    # main window logic
    ###################
    def _enable_global_input(self):
        for action in self._was_enabled:
            action.setEnabled(True)

    def _disable_global_input(self):
        self._was_enabled = []
        for action in self.all_actions:
            if action.isEnabled():
                self._was_enabled.append(action)
            action.setEnabled(False)

    def _create_document_base(self):
        self.show_create_document_base_widget()

    def _load_document_base_from_bson(self):
        path = str(QFileDialog.getOpenFileName(self, "Choose a document collection .bson file!")[0])
        if path != "":
            self._disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_document_base_from_bson.emit(path)

    def _save_document_base_to_bson(self):
        if self.document_base is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the document collection .bson file!")[0])
            if path != "":
                self._disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_document_base_to_bson.emit(path, self.document_base)

    def _save_table_to_csv(self):
        if self.document_base is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the table .csv file!")[0])
            if path != "":
                self._disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_table_to_csv.emit(path, self.document_base)

    def _load_default_preprocessing_phase(self):
        self._disable_global_input()
        # noinspection PyUnresolvedReferences
        self.load_default_preprocessing_phase.emit()

    def _load_preprocessing_phase_from_config(self):
        path = str(QFileDialog.getOpenFileName(self, "Choose a configuration .json file!")[0])
        if path != "":
            self._disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_preprocessing_phase_from_config.emit(path)

    def _save_preprocessing_phase_to_config(self):
        if self.preprocessing_phase is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the configuration .json file!")[0])
            if path != "":
                self._disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_preprocessing_phase_to_config.emit(path, self.preprocessing_phase)

    def _load_default_matching_phase(self):
        self._disable_global_input()
        # noinspection PyUnresolvedReferences
        self.load_default_matching_phase.emit()

    def _load_matching_phase_from_config(self):
        path = str(QFileDialog.getOpenFileName(self, "Choose a configuration .json file!")[0])
        if path != "":
            self._disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_matching_phase_from_config.emit(path)

    def _save_matching_phase_to_config(self):
        if self.matching_phase is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the configuration .json file!")[0])
            if path != "":
                self._disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_matching_phase_to_config.emit(path, self.matching_phase)

    def _run_preprocessing_phase(self):
        if self.document_base is not None and self.preprocessing_phase is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self._disable_global_input()

            # noinspection PyUnresolvedReferences
            self.run_preprocessing_phase.emit(self.document_base, self.preprocessing_phase, self.statistics)

    def _run_matching_phase(self):
        if self.document_base is not None and self.matching_phase is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self.show_interactive_matching_widget()
            self._disable_global_input()

            # noinspection PyUnresolvedReferences
            self.run_matching_phase.emit(self.document_base, self.matching_phase, self.statistics)

    def _enable_collect_statistics(self):
        self.collect_statistics = True
        self.enable_collect_statistics_action.setEnabled(False)
        self.disable_collect_statistics_action.setEnabled(True)

    def _disable_collect_statistics(self):
        self.collect_statistics = False
        self.disable_collect_statistics_action.setEnabled(False)
        self.enable_collect_statistics_action.setEnabled(True)

    def _save_statistics_to_json(self):
        path = str(QFileDialog.getSaveFileName(self, "Choose where to save the statistics .json file!")[0])
        if path != "":
            self._disable_global_input()
            # noinspection PyUnresolvedReferences
            self.save_statistics_to_json.emit(path, self.statistics)

    def show_document_base_viewer_widget(self):
        if self.document_base_viewer_widget.isHidden():
            self.start_menu_widget.hide()
            self.interactive_matching_widget.hide()
            self.create_document_base_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.interactive_matching_widget)
            self.central_widget_layout.removeWidget(self.create_document_base_widget)
            self.central_widget_layout.addWidget(self.document_base_viewer_widget)
            self.document_base_viewer_widget.show()
            self.central_widget_layout.update()

    def show_interactive_matching_widget(self):
        if self.interactive_matching_widget.isHidden():
            self.start_menu_widget.hide()
            self.document_base_viewer_widget.hide()
            self.create_document_base_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.document_base_viewer_widget)
            self.central_widget_layout.removeWidget(self.create_document_base_widget)
            self.central_widget_layout.addWidget(self.interactive_matching_widget)
            self.interactive_matching_widget.show()
            self.central_widget_layout.update()

    def show_create_document_base_widget(self):
        if self.create_document_base_widget.isHidden():
            self.start_menu_widget.hide()
            self.document_base_viewer_widget.hide()
            self.interactive_matching_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.document_base_viewer_widget)
            self.central_widget_layout.removeWidget(self.interactive_matching_widget)
            self.central_widget_layout.addWidget(self.create_document_base_widget)
            self.create_document_base_widget.show()
            self.central_widget_layout.update()

    def give_feedback(self, feedback):
        self.api.feedback = feedback
        self.feedback_cond.wakeAll()

    def create_new_document_base(self, path, attribute_names):
        self._disable_global_input()
        # noinspection PyUnresolvedReferences
        self.create_document_base.emit(path, attribute_names)

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setWindowTitle("ASET")

        self.document_base = None
        self.preprocessing_phase = None
        self.matching_phase = None
        self.statistics = Statistics(True)
        self.collect_statistics = True

        # set up the api_thread and api and connect slots and signals
        self.feedback_mutex = QMutex()
        self.feedback_cond = QWaitCondition()
        self.api = ASETAPI(self.feedback_mutex, self.feedback_cond)
        self.api_thread = QThread()
        self.api.moveToThread(self.api_thread)
        self._connect_slots_and_signals()
        self.api_thread.start()

        # set up the status bar
        self.status_bar = self.statusBar()
        self.status_bar.setFont(STATUS_BAR_FONT)

        self.status_widget = QWidget(self.status_bar)
        self.status_widget_layout = QHBoxLayout(self.status_widget)
        self.status_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.status_widget_message = QLabel()
        self.status_widget_message.setFont(STATUS_BAR_FONT)
        self.status_widget_message.setMinimumWidth(10)
        self.status_widget_layout.addWidget(self.status_widget_message)
        self.status_widget_progress = QProgressBar()
        self.status_widget_progress.setMinimumWidth(10)
        self.status_widget_progress.setMaximumWidth(100)
        self.status_widget_progress.setTextVisible(False)
        self.status_widget_layout.addWidget(self.status_widget_progress)
        self.status_bar.addPermanentWidget(self.status_widget)

        # set up the actions
        self.all_actions = []
        self._was_enabled = []

        self.exit_action = QAction("&Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.setStatusTip("Exit application.")
        self.exit_action.triggered.connect(QApplication.instance().quit)
        self.all_actions.append(self.exit_action)

        self.create_document_base_action = QAction("&Create document base", self)
        self.create_document_base_action.setStatusTip(
            "Create a new document base from a collection of .txt files and a list of attribute names."
        )
        self.create_document_base_action.triggered.connect(self._create_document_base)
        self.all_actions.append(self.create_document_base_action)

        self.load_document_base_from_bson_action = QAction("&Load document base", self)
        self.load_document_base_from_bson_action.setShortcut("Ctrl+O")
        self.load_document_base_from_bson_action.setStatusTip("Load the document base from a .bson file.")
        self.load_document_base_from_bson_action.triggered.connect(self._load_document_base_from_bson)
        self.all_actions.append(self.load_document_base_from_bson_action)

        self.save_document_base_to_bson_action = QAction("&Save document base", self)
        self.save_document_base_to_bson_action.setShortcut("Ctrl+S")
        self.save_document_base_to_bson_action.setStatusTip("Save the document base in a .bson file.")
        self.save_document_base_to_bson_action.triggered.connect(self._save_document_base_to_bson)
        self.save_document_base_to_bson_action.setEnabled(False)
        self.all_actions.append(self.save_document_base_to_bson_action)

        self.save_table_to_csv_action = QAction("&Save table", self)
        self.save_table_to_csv_action.setStatusTip("Save the table to a .csv file.")
        self.save_table_to_csv_action.triggered.connect(self._save_table_to_csv)
        self.save_table_to_csv_action.setEnabled(False)
        self.all_actions.append(self.save_table_to_csv_action)

        self.load_default_preprocessing_phase_action = QAction("&Load default preprocessing phase", self)
        self.load_default_preprocessing_phase_action.setStatusTip("Load the default preprocessing phase.")
        self.load_default_preprocessing_phase_action.triggered.connect(self._load_default_preprocessing_phase)
        self.all_actions.append(self.load_default_preprocessing_phase_action)

        self.load_preprocessing_phase_from_config_action = QAction("&Load preprocessing phase", self)
        self.load_preprocessing_phase_from_config_action.setStatusTip(
            "Load a preprocessing phase from a .json configuration file."
        )
        self.load_preprocessing_phase_from_config_action.triggered.connect(self._load_preprocessing_phase_from_config)
        self.all_actions.append(self.load_preprocessing_phase_from_config_action)

        self.save_preprocessing_phase_to_config_action = QAction("&Save preprocessing phase", self)
        self.save_preprocessing_phase_to_config_action.setStatusTip(
            "Save the preprocessing phase in a .json configuration file."
        )
        self.save_preprocessing_phase_to_config_action.triggered.connect(self._save_preprocessing_phase_to_config)
        self.save_preprocessing_phase_to_config_action.setEnabled(False)
        self.all_actions.append(self.save_preprocessing_phase_to_config_action)

        self.run_preprocessing_phase_action = QAction("Run preprocessing phase", self)
        self.run_preprocessing_phase_action.setStatusTip("Run the preprocessing phase on the document collection.")
        self.run_preprocessing_phase_action.triggered.connect(self._run_preprocessing_phase)
        self.run_preprocessing_phase_action.setEnabled(False)
        self.all_actions.append(self.run_preprocessing_phase_action)

        self.load_default_matching_phase_action = QAction("&Load default matching phase", self)
        self.load_default_matching_phase_action.setStatusTip("Load the default matching phase.")
        self.load_default_matching_phase_action.triggered.connect(self._load_default_matching_phase)
        self.all_actions.append(self.load_default_matching_phase_action)

        self.load_matching_phase_from_config_action = QAction("&Load matching phase", self)
        self.load_matching_phase_from_config_action.setStatusTip(
            "Load a matching phase from a .json configuration file."
        )
        self.load_matching_phase_from_config_action.triggered.connect(self._load_matching_phase_from_config)
        self.all_actions.append(self.load_matching_phase_from_config_action)

        self.save_matching_phase_to_config_action = QAction("&Save matching phase", self)
        self.save_matching_phase_to_config_action.setStatusTip("Save the matching phase in a .json configuration file.")
        self.save_matching_phase_to_config_action.triggered.connect(self._save_matching_phase_to_config)
        self.save_matching_phase_to_config_action.setEnabled(False)
        self.all_actions.append(self.save_matching_phase_to_config_action)

        self.run_matching_phase_action = QAction("Run matching phase", self)
        self.run_matching_phase_action.setStatusTip("Run the matching phase on the document collection.")
        self.run_matching_phase_action.triggered.connect(self._run_matching_phase)
        self.run_matching_phase_action.setEnabled(False)
        self.all_actions.append(self.run_matching_phase_action)

        self.enable_collect_statistics_action = QAction("&Enable statistics", self)
        self.enable_collect_statistics_action.setStatusTip("Enable collecting statistics.")
        self.enable_collect_statistics_action.triggered.connect(self._enable_collect_statistics)
        self.enable_collect_statistics_action.setEnabled(False)
        self.all_actions.append(self.enable_collect_statistics_action)

        self.disable_collect_statistics_action = QAction("&Disable statistics", self)
        self.disable_collect_statistics_action.setStatusTip("Disable collecting statistics.")
        self.disable_collect_statistics_action.triggered.connect(self._disable_collect_statistics)
        self.all_actions.append(self.disable_collect_statistics_action)

        self.save_statistics_to_json_action = QAction("&Save statistics", self)
        self.save_statistics_to_json_action.setStatusTip("Save the statistics to a .json file.")
        self.save_statistics_to_json_action.triggered.connect(self._save_statistics_to_json)
        self.save_statistics_to_json_action.setEnabled(False)
        self.all_actions.append(self.save_statistics_to_json_action)

        # set up the menu bar
        self.menubar = self.menuBar()
        self.menubar.setFont(MENU_FONT)

        self.file_menu = self.menubar.addMenu("&File")
        self.file_menu.setFont(MENU_FONT)
        self.file_menu.addAction(self.exit_action)

        self.document_base_menu = self.menubar.addMenu("&Document Base")
        self.document_base_menu.setFont(MENU_FONT)
        self.document_base_menu.addAction(self.create_document_base_action)
        self.document_base_menu.addAction(self.load_document_base_from_bson_action)
        self.document_base_menu.addAction(self.save_document_base_to_bson_action)
        self.document_base_menu.addAction(self.save_table_to_csv_action)

        self.preprocessing_menu = self.menubar.addMenu("&Preprocessing")
        self.preprocessing_menu.setFont(MENU_FONT)
        self.preprocessing_menu.addAction(self.load_default_preprocessing_phase_action)
        self.preprocessing_menu.addAction(self.load_preprocessing_phase_from_config_action)
        self.preprocessing_menu.addAction(self.save_preprocessing_phase_to_config_action)
        self.preprocessing_menu.addAction(self.run_preprocessing_phase_action)

        self.matching_menu = self.menubar.addMenu("&Matching")
        self.matching_menu.setFont(MENU_FONT)
        self.matching_menu.addAction(self.load_default_matching_phase_action)
        self.matching_menu.addAction(self.load_matching_phase_from_config_action)
        self.matching_menu.addAction(self.save_matching_phase_to_config_action)
        self.matching_menu.addAction(self.run_matching_phase_action)

        self.statistics_menu = self.menubar.addMenu("&Statistics")
        self.statistics_menu.setFont(MENU_FONT)
        self.statistics_menu.addAction(self.enable_collect_statistics_action)
        self.statistics_menu.addAction(self.disable_collect_statistics_action)
        self.statistics_menu.addAction(self.save_statistics_to_json_action)

        # start menu
        self.start_menu_widget = QWidget()
        self.start_menu_layout = QVBoxLayout(self.start_menu_widget)
        self.start_menu_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_layout.setSpacing(20)
        self.start_menu_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.start_menu_widget.setMaximumWidth(400)

        self.start_menu_header = QLabel("Welcome to ASET!")
        self.start_menu_header.setFont(HEADER_FONT)
        self.start_menu_layout.addWidget(self.start_menu_header)

        self.create_document_base_button = QPushButton("Create a new Document Base")
        self.create_document_base_button.setFont(BUTTON_FONT)
        self.create_document_base_button.clicked.connect(self._create_document_base)
        self.start_menu_layout.addWidget(self.create_document_base_button)

        self.load_document_base_button = QPushButton("Load an existing Document Base")
        self.load_document_base_button.setFont(BUTTON_FONT)
        self.load_document_base_button.clicked.connect(self._load_document_base_from_bson)
        self.start_menu_layout.addWidget(self.load_document_base_button)

        # main UI
        self.central_widget = QWidget(self)
        self.central_widget_layout = QHBoxLayout()
        self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.central_widget.setLayout(self.central_widget_layout)
        self.setCentralWidget(self.central_widget)

        self.create_document_base_widget = CreateDocumentBaseWidget(self)
        self.document_base_viewer_widget = DocumentBaseViewerWidget(self)
        self.interactive_matching_widget = InteractiveMatchingWidget(self)

        self.create_document_base_widget.hide()
        self.document_base_viewer_widget.hide()
        self.interactive_matching_widget.hide()
        self.central_widget_layout.addWidget(self.start_menu_widget)
        self.central_widget_layout.update()

        self.resize(1400, 800)
        self.show()

        logger.info("Initialized MainWindow.")
