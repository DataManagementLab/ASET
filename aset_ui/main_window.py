import logging

from PyQt6.QtCore import QMutex, Qt, QThread, QWaitCondition, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, \
    QVBoxLayout, QWidget, QInputDialog

from aset.data.data import ASETDocumentBase
from aset.matching.phase import BaseMatchingPhase
from aset.preprocessing.phase import PreprocessingPhase
from aset.statistics import Statistics
from aset_ui.aset_api import ASETAPI
from aset_ui.common import HEADER_FONT, MENU_FONT, STATUS_BAR_FONT, SUBHEADER_FONT, LABEL_FONT
from aset_ui.document_base import DocumentBaseCreatorWidget, DocumentBaseViewerWidget
from aset_ui.interactive_matching import InteractiveMatchingWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    ################################
    # signals (aset ui --> aset api)
    ################################
    create_document_base = pyqtSignal(str, list)
    add_attribute = pyqtSignal(str, ASETDocumentBase)
    remove_attribute = pyqtSignal(str, ASETDocumentBase)
    forget_matches_for_attribute = pyqtSignal(str, ASETDocumentBase)
    load_document_base_from_bson = pyqtSignal(str)
    save_document_base_to_bson = pyqtSignal(str, ASETDocumentBase)
    save_table_to_csv = pyqtSignal(str, ASETDocumentBase)
    forget_matches = pyqtSignal(ASETDocumentBase)
    load_preprocessing_phase_from_config = pyqtSignal(str)
    save_preprocessing_phase_to_config = pyqtSignal(str, PreprocessingPhase)
    load_matching_phase_from_config = pyqtSignal(str)
    save_matching_phase_to_config = pyqtSignal(str, BaseMatchingPhase)
    run_preprocessing_phase = pyqtSignal(ASETDocumentBase, PreprocessingPhase, Statistics)
    run_matching_phase = pyqtSignal(ASETDocumentBase, BaseMatchingPhase, Statistics)
    save_statistics_to_json = pyqtSignal(str, Statistics)
    load_and_run_default_preprocessing_phase = pyqtSignal(ASETDocumentBase, Statistics)
    load_and_run_default_matching_phase = pyqtSignal(ASETDocumentBase, Statistics)

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
        self.enable_global_input()

    @pyqtSlot(str)
    def error(self, message):
        logger.debug("Called slot 'error'.")

        self.status_widget_message.setText(message)
        self.status_widget_progress.setRange(0, 100)
        self.status_widget_progress.setValue(0)

        self.show_document_base_viewer_widget()
        self.enable_global_input()

    @pyqtSlot(ASETDocumentBase)
    def document_base_to_ui(self, document_base):
        logger.debug("Called slot 'document_base_to_ui'.")

        self.document_base = document_base
        self.document_base_viewer_widget.update_document_base(self.document_base)

        self._was_enabled.append(self.save_document_base_to_bson_action)
        self._was_enabled.append(self.save_table_to_csv_action)
        self._was_enabled.append(self.add_attribute_action)
        self._was_enabled.append(self.remove_attribute_action)
        self._was_enabled.append(self.forget_matches_for_attribute_action)
        self._was_enabled.append(self.forget_matches_action)
        self._was_enabled.append(self.load_and_run_default_preprocessing_phase_action)
        self._was_enabled.append(self.load_and_run_default_matching_phase_action)
        if self.preprocessing_phase is not None:
            self._was_enabled.append(self.run_preprocessing_phase_action)
        if self.matching_phase is not None:
            self._was_enabled.append(self.run_matching_phase_action)

    @pyqtSlot(PreprocessingPhase)
    def preprocessing_phase_to_ui(self, preprocessing_phase):
        logger.debug("Called slot 'preprocessing_phase_to_ui'.")

        self.preprocessing_phase = preprocessing_phase

        self._was_enabled.append(self.save_preprocessing_phase_to_config_action)
        if self.document_base is not None:
            self._was_enabled.append(self.run_preprocessing_phase_action)

    @pyqtSlot(BaseMatchingPhase)
    def matching_phase_to_ui(self, matching_phase):
        logger.debug("Called slot 'matching_phase_to_ui'.")

        self.matching_phase = matching_phase

        self._was_enabled.append(self.save_matching_phase_to_config_action)
        if self.document_base is not None:
            self._was_enabled.append(self.run_preprocessing_phase_action)

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
        self.add_attribute.connect(self.api.add_attribute)
        self.remove_attribute.connect(self.api.remove_attribute)
        self.forget_matches_for_attribute.connect(self.api.forget_matches_for_attribute)
        self.load_document_base_from_bson.connect(self.api.load_document_base_from_bson)
        self.save_document_base_to_bson.connect(self.api.save_document_base_to_bson)
        self.save_table_to_csv.connect(self.api.save_table_to_csv)
        self.forget_matches.connect(self.api.forget_matches)
        self.load_preprocessing_phase_from_config.connect(self.api.load_preprocessing_phase_from_config)
        self.save_preprocessing_phase_to_config.connect(self.api.save_preprocessing_phase_to_config)
        self.load_matching_phase_from_config.connect(self.api.load_matching_phase_from_config)
        self.save_matching_phase_to_config.connect(self.api.save_matching_phase_to_config)
        self.run_preprocessing_phase.connect(self.api.run_preprocessing_phase)
        self.run_matching_phase.connect(self.api.run_matching_phase)
        self.save_statistics_to_json.connect(self.api.save_statistics_to_json)
        self.load_and_run_default_preprocessing_phase.connect(self.api.load_and_run_default_preprocessing_phase)
        self.load_and_run_default_matching_phase.connect(self.api.load_and_run_default_matching_phase)

        self.api.status.connect(self.status)
        self.api.finished.connect(self.finished)
        self.api.error.connect(self.error)
        self.api.document_base_to_ui.connect(self.document_base_to_ui)
        self.api.preprocessing_phase_to_ui.connect(self.preprocessing_phase_to_ui)
        self.api.matching_phase_to_ui.connect(self.matching_phase_to_ui)
        self.api.statistics_to_ui.connect(self.statistics_to_ui)
        self.api.feedback_request_to_ui.connect(self.feedback_request_to_ui)

    #######
    # tasks
    #######
    def load_document_base_from_bson_task(self):
        logger.info("Execute task 'load_document_base_from_bson_task'.")

        path, ok = QFileDialog.getOpenFileName(self, "Choose a document collection .bson file!")
        if ok:
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_document_base_from_bson.emit(str(path))

    def save_document_base_to_bson_task(self):
        logger.info("Execute task 'save_document_base_to_bson_task'.")

        if self.document_base is not None:
            path, ok = QFileDialog.getSaveFileName(self, "Choose where to save the document collection .bson file!")
            if ok:
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_document_base_to_bson.emit(str(path), self.document_base)

    def add_attribute_task(self):
        logger.info("Execute task 'add_attribute_task'.")

        if self.document_base is not None:
            name, ok = QInputDialog.getText(self, "Create Attribute", "Attribute name:")
            if ok:
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.add_attribute.emit(str(name), self.document_base)

    def remove_attribute_task(self):
        logger.info("Execute task 'remove_attribute_task'.")

        if self.document_base is not None:
            name, ok = QInputDialog.getText(self, "Remove Attribute", "Attribute name:")
            if ok:
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.remove_attribute.emit(str(name), self.document_base)

    def remove_attribute_with_given_name_task(self, attribute_name):
        logger.info("Execute task 'remove_attribute_with_given_name_task'.")

        if self.document_base is not None:
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.remove_attribute.emit(str(attribute_name), self.document_base)

    def forget_matches_for_attribute_task(self):
        logger.info("Execute task 'forget_matches_for_attribute_task'.")

        if self.document_base is not None:
            name, ok = QInputDialog.getText(self, "Forget Matches for Attribute", "Attribute name:")
            if ok:
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.forget_matches_for_attribute.emit(str(name), self.document_base)

    def forget_matches_for_attribute_with_given_name_task(self, attribute_name):
        logger.info("Execute task 'forget_matches_for_attribute_with_given_name_task'.")

        if self.document_base is not None:
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.forget_matches_for_attribute.emit(attribute_name, self.document_base)

    def forget_matches_task(self):
        logger.info("Execute task 'forget_matches_task'.")

        if self.document_base is not None:
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.forget_matches.emit(self.document_base)

    def enable_collect_statistics_task(self):
        logger.info("Execute task 'task_enable_collect_statistics'.")

        self.collect_statistics = True
        self.enable_collect_statistics_action.setEnabled(False)
        self.disable_collect_statistics_action.setEnabled(True)

    def disable_collect_statistics_task(self):
        logger.info("Execute task 'disable_collect_statistics_task'.")

        self.collect_statistics = False
        self.disable_collect_statistics_action.setEnabled(False)
        self.enable_collect_statistics_action.setEnabled(True)

    def save_statistics_to_json_task(self):
        logger.info("Execute task 'save_statistics_to_json_task'.")

        if self.statistics is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the statistics .json file!")[0])
            if path != "":
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_statistics_to_json.emit(path, self.statistics)

    def show_document_base_creator_widget_task(self):
        logger.info("Execute task 'show_document_base_creator_widget_task'.")

        self.disable_global_input()
        self.document_base_creator_widget.enable_input()
        self.document_base_creator_widget.initialize_for_new_document_base()
        self.show_document_base_creator_widget()
        self.document_base_creator_widget.path.setFocus()

    def create_document_base_task(self, path, attribute_names):
        logger.info("Execute task 'create_document_base_task'.")

        self.disable_global_input()
        # noinspection PyUnresolvedReferences
        self.create_document_base.emit(path, attribute_names)

    def save_table_to_csv_task(self):
        logger.info("Execute task 'save_table_to_csv_task'.")

        if self.document_base is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the table .csv file!")[0])
            if path != "":
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_table_to_csv.emit(path, self.document_base)

    def load_preprocessing_phase_from_config_task(self):
        logger.info("Execute task 'load_preprocessing_phase_from_config_task'.")

        path = str(QFileDialog.getOpenFileName(self, "Choose a configuration .json file!")[0])
        if path != "":
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_preprocessing_phase_from_config.emit(path)

    def save_preprocessing_phase_to_config_task(self):
        logger.info("Execute task 'save_preprocessing_phase_to_config_task'.")

        if self.preprocessing_phase is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the configuration .json file!")[0])
            if path != "":
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_preprocessing_phase_to_config.emit(path, self.preprocessing_phase)

    def load_matching_phase_from_config_task(self):
        logger.info("Execute task 'load_matching_phase_from_config_task'.")

        path = str(QFileDialog.getOpenFileName(self, "Choose a configuration .json file!")[0])
        if path != "":
            self.disable_global_input()
            # noinspection PyUnresolvedReferences
            self.load_matching_phase_from_config.emit(path)

    def save_matching_phase_to_config_task(self):
        logger.info("Execute task 'save_matching_phase_to_config_task'.")

        if self.matching_phase is not None:
            path = str(QFileDialog.getSaveFileName(self, "Choose where to save the configuration .json file!")[0])
            if path != "":
                self.disable_global_input()
                # noinspection PyUnresolvedReferences
                self.save_matching_phase_to_config.emit(path, self.matching_phase)

    def run_preprocessing_phase_task(self):
        logger.info("Execute task 'run_preprocessing_phase_task'.")

        if self.document_base is not None and self.preprocessing_phase is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self.disable_global_input()

            # noinspection PyUnresolvedReferences
            self.run_preprocessing_phase.emit(self.document_base, self.preprocessing_phase, self.statistics)

    def run_matching_phase_task(self):
        logger.info("Execute task 'run_matching_phase_task'.")

        if self.document_base is not None and self.matching_phase is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self.disable_global_input()
            self.interactive_matching_widget.enable_input()
            self.show_interactive_matching_widget()

            # noinspection PyUnresolvedReferences
            self.run_matching_phase.emit(self.document_base, self.matching_phase, self.statistics)

    def give_feedback_task(self, feedback):
        logger.info("Execute task 'give_feedback_task'.")

        self.api.feedback = feedback
        self.feedback_cond.wakeAll()

    def load_and_run_default_preprocessing_phase_task(self):
        logger.info("Execute task 'load_and_run_default_preprocessing_phase_task'.")

        if self.document_base is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self.disable_global_input()

            # noinspection PyUnresolvedReferences
            self.load_and_run_default_preprocessing_phase.emit(self.document_base, self.statistics)

    def load_and_run_default_matching_phase_task(self):
        logger.info("Execute task 'load_and_run_default_matching_phase_task'.")

        if self.document_base is not None:
            self.statistics = Statistics(self.collect_statistics)
            self.save_statistics_to_json_action.setEnabled(self.collect_statistics)

            self.disable_global_input()
            self.interactive_matching_widget.enable_input()
            self.show_interactive_matching_widget()

            # noinspection PyUnresolvedReferences
            self.load_and_run_default_matching_phase.emit(self.document_base, self.statistics)

    ##################
    # controller logic
    ##################
    def enable_global_input(self):
        for action in self._was_enabled:
            action.setEnabled(True)

        self.document_base_creator_widget.enable_input()
        self.document_base_viewer_widget.enable_input()
        self.interactive_matching_widget.enable_input()
        self._was_enabled = []

    def disable_global_input(self):
        for action in self._all_actions:
            if action.isEnabled():
                self._was_enabled.append(action)
            action.setEnabled(False)

        self.document_base_creator_widget.disable_input()
        self.document_base_viewer_widget.disable_input()
        self.interactive_matching_widget.disable_input()

    def show_document_base_viewer_widget(self):
        if self.document_base_viewer_widget.isHidden():
            self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.start_menu_widget.hide()
            self.interactive_matching_widget.hide()
            self.document_base_creator_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.interactive_matching_widget)
            self.central_widget_layout.removeWidget(self.document_base_creator_widget)
            self.central_widget_layout.addWidget(self.document_base_viewer_widget)
            self.document_base_viewer_widget.show()
            self.central_widget_layout.update()

    def show_interactive_matching_widget(self):
        if self.interactive_matching_widget.isHidden():
            self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.start_menu_widget.hide()
            self.document_base_viewer_widget.hide()
            self.document_base_creator_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.document_base_viewer_widget)
            self.central_widget_layout.removeWidget(self.document_base_creator_widget)
            self.central_widget_layout.addWidget(self.interactive_matching_widget)
            self.interactive_matching_widget.show()
            self.central_widget_layout.update()

    def show_document_base_creator_widget(self):
        if self.document_base_creator_widget.isHidden():
            self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.start_menu_widget.hide()
            self.document_base_viewer_widget.hide()
            self.interactive_matching_widget.hide()

            self.central_widget_layout.removeWidget(self.start_menu_widget)
            self.central_widget_layout.removeWidget(self.document_base_viewer_widget)
            self.central_widget_layout.removeWidget(self.interactive_matching_widget)
            self.central_widget_layout.addWidget(self.document_base_creator_widget)
            self.document_base_creator_widget.show()
            self.central_widget_layout.update()

    def show_start_menu_widget(self):
        if self.start_menu_widget.isHidden():
            self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.document_base_viewer_widget.hide()
            self.document_base_creator_widget.hide()
            self.interactive_matching_widget.hide()

            self.central_widget_layout.removeWidget(self.document_base_viewer_widget)
            self.central_widget_layout.removeWidget(self.document_base_creator_widget)
            self.central_widget_layout.removeWidget(self.interactive_matching_widget)
            self.central_widget_layout.addWidget(self.start_menu_widget)
            self.start_menu_widget.show()
            self.central_widget_layout.update()

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
        self.status_widget_progress.setMaximumWidth(200)
        self.status_widget_progress.setTextVisible(False)
        self.status_widget_progress.setMaximumHeight(20)
        self.status_widget_layout.addWidget(self.status_widget_progress)
        self.status_bar.addPermanentWidget(self.status_widget)

        # set up the actions
        self._all_actions = []
        self._was_enabled = []

        self.exit_action = QAction("&Exit", self)
        self.exit_action.setIcon(QIcon("aset_ui/resources/leave.svg"))
        self.exit_action.setStatusTip("Exit the application.")
        self.exit_action.triggered.connect(QApplication.instance().quit)
        self._all_actions.append(self.exit_action)

        self.show_document_base_creator_widget_action = QAction("&Create new document base", self)
        self.show_document_base_creator_widget_action.setIcon(QIcon("aset_ui/resources/two_documents.svg"))
        self.show_document_base_creator_widget_action.setStatusTip(
            "Create a new document base from a collection of .txt files and a list of attribute names."
        )
        self.show_document_base_creator_widget_action.triggered.connect(self.show_document_base_creator_widget_task)
        self._all_actions.append(self.show_document_base_creator_widget_action)

        self.add_attribute_action = QAction("&Add attribute", self)
        self.add_attribute_action.setIcon(QIcon("aset_ui/resources/plus.svg"))
        self.add_attribute_action.setStatusTip("Add a new attribute to the document base.")
        self.add_attribute_action.triggered.connect(self.add_attribute_task)
        self.add_attribute_action.setEnabled(False)
        self._all_actions.append(self.add_attribute_action)

        self.remove_attribute_action = QAction("&Remove attribute", self)
        self.remove_attribute_action.setIcon(QIcon("aset_ui/resources/trash.svg"))
        self.remove_attribute_action.setStatusTip("Remove an attribute from the document base.")
        self.remove_attribute_action.triggered.connect(self.remove_attribute_task)
        self.remove_attribute_action.setEnabled(False)
        self._all_actions.append(self.remove_attribute_action)

        self.forget_matches_for_attribute_action = QAction("&Forget matches for attribute", self)
        self.forget_matches_for_attribute_action.setIcon(QIcon("aset_ui/resources/redo.svg"))
        self.forget_matches_for_attribute_action.setStatusTip("Forget the matches for a single attribute.")
        self.forget_matches_for_attribute_action.triggered.connect(self.forget_matches_for_attribute_task)
        self.forget_matches_for_attribute_action.setEnabled(False)
        self._all_actions.append(self.forget_matches_for_attribute_action)

        self.load_document_base_from_bson_action = QAction("&Load document base", self)
        self.load_document_base_from_bson_action.setIcon(QIcon("aset_ui/resources/folder.svg"))
        self.load_document_base_from_bson_action.setStatusTip("Load an existing document base from a .bson file.")
        self.load_document_base_from_bson_action.triggered.connect(self.load_document_base_from_bson_task)
        self._all_actions.append(self.load_document_base_from_bson_action)

        self.save_document_base_to_bson_action = QAction("&Save document base", self)
        self.save_document_base_to_bson_action.setIcon(QIcon("aset_ui/resources/save.svg"))
        self.save_document_base_to_bson_action.setStatusTip("Save the document base in a .bson file.")
        self.save_document_base_to_bson_action.triggered.connect(self.save_document_base_to_bson_task)
        self.save_document_base_to_bson_action.setEnabled(False)
        self._all_actions.append(self.save_document_base_to_bson_action)

        self.save_table_to_csv_action = QAction("&Export table to CSV", self)
        self.save_table_to_csv_action.setIcon(QIcon("aset_ui/resources/table.svg"))
        self.save_table_to_csv_action.setStatusTip("Save the table to a .csv file.")
        self.save_table_to_csv_action.triggered.connect(self.save_table_to_csv_task)
        self.save_table_to_csv_action.setEnabled(False)
        self._all_actions.append(self.save_table_to_csv_action)

        self.forget_matches_action = QAction("&Forget all matches", self)
        self.forget_matches_action.setIcon(QIcon("aset_ui/resources/redo.svg"))
        self.forget_matches_action.setStatusTip("Forget the matches for all attributes.")
        self.forget_matches_action.triggered.connect(self.forget_matches_task)
        self.forget_matches_action.setEnabled(False)
        self._all_actions.append(self.forget_matches_action)

        self.load_and_run_default_preprocessing_phase_action = QAction(
            "&Load and run default preprocessing phase", self
        )
        self.load_and_run_default_preprocessing_phase_action.setStatusTip(
            "Load the default preprocessing phase and run it on the document collection."
        )
        self.load_and_run_default_preprocessing_phase_action.setIcon(QIcon("aset_ui/resources/run_run.svg"))
        self.load_and_run_default_preprocessing_phase_action.setDisabled(True)
        self.load_and_run_default_preprocessing_phase_action.triggered.connect(
            self.load_and_run_default_preprocessing_phase_task
        )
        self._all_actions.append(self.load_and_run_default_preprocessing_phase_action)

        self.load_preprocessing_phase_from_config_action = QAction("&Load preprocessing phase", self)
        self.load_preprocessing_phase_from_config_action.setStatusTip(
            "Load a preprocessing phase from a .json configuration file."
        )
        self.load_preprocessing_phase_from_config_action.triggered.connect(
            self.load_preprocessing_phase_from_config_task
        )
        self._all_actions.append(self.load_preprocessing_phase_from_config_action)

        self.save_preprocessing_phase_to_config_action = QAction("&Save preprocessing phase", self)
        self.save_preprocessing_phase_to_config_action.setStatusTip(
            "Save the preprocessing phase in a .json configuration file."
        )
        self.save_preprocessing_phase_to_config_action.triggered.connect(self.save_preprocessing_phase_to_config_task)
        self.save_preprocessing_phase_to_config_action.setEnabled(False)
        self._all_actions.append(self.save_preprocessing_phase_to_config_action)

        self.run_preprocessing_phase_action = QAction("Run preprocessing phase", self)
        self.run_preprocessing_phase_action.setIcon(QIcon("aset_ui/resources/run.svg"))
        self.run_preprocessing_phase_action.setStatusTip("Run the preprocessing phase on the document collection.")
        self.run_preprocessing_phase_action.triggered.connect(self.run_preprocessing_phase_task)
        self.run_preprocessing_phase_action.setEnabled(False)
        self._all_actions.append(self.run_preprocessing_phase_action)

        self.load_and_run_default_matching_phase_action = QAction(
            "&Load and run default matching phase", self
        )
        self.load_and_run_default_matching_phase_action.setStatusTip(
            "Load the default matching phase and run it on the document collection."
        )
        self.load_and_run_default_matching_phase_action.setIcon(QIcon("aset_ui/resources/run_run.svg"))
        self.load_and_run_default_matching_phase_action.setDisabled(True)
        self.load_and_run_default_matching_phase_action.triggered.connect(
            self.load_and_run_default_preprocessing_phase_task
        )
        self._all_actions.append(self.load_and_run_default_matching_phase_action)

        self.load_matching_phase_from_config_action = QAction("&Load matching phase", self)
        self.load_matching_phase_from_config_action.setStatusTip(
            "Load a matching phase from a .json configuration file."
        )
        self.load_matching_phase_from_config_action.triggered.connect(self.load_matching_phase_from_config_task)
        self._all_actions.append(self.load_matching_phase_from_config_action)

        self.save_matching_phase_to_config_action = QAction("&Save matching phase", self)
        self.save_matching_phase_to_config_action.setStatusTip("Save the matching phase in a .json configuration file.")
        self.save_matching_phase_to_config_action.triggered.connect(self.save_matching_phase_to_config_task)
        self.save_matching_phase_to_config_action.setEnabled(False)
        self._all_actions.append(self.save_matching_phase_to_config_action)

        self.run_matching_phase_action = QAction("Run matching phase", self)
        self.run_matching_phase_action.setIcon(QIcon("aset_ui/resources/run.svg"))
        self.run_matching_phase_action.setStatusTip("Run the matching phase on the document collection.")
        self.run_matching_phase_action.triggered.connect(self.run_matching_phase_task)
        self.run_matching_phase_action.setEnabled(False)
        self._all_actions.append(self.run_matching_phase_action)

        self.enable_collect_statistics_action = QAction("&Enable statistics", self)
        self.enable_collect_statistics_action.setIcon(QIcon("aset_ui/resources/statistics.svg"))
        self.enable_collect_statistics_action.setStatusTip("Enable collecting statistics.")
        self.enable_collect_statistics_action.triggered.connect(self.enable_collect_statistics_task)
        self.enable_collect_statistics_action.setEnabled(False)
        self._all_actions.append(self.enable_collect_statistics_action)

        self.disable_collect_statistics_action = QAction("&Disable statistics", self)
        self.disable_collect_statistics_action.setIcon(QIcon("aset_ui/resources/statistics_incorrect.svg"))
        self.disable_collect_statistics_action.setStatusTip("Disable collecting statistics.")
        self.disable_collect_statistics_action.triggered.connect(self.disable_collect_statistics_task)
        self._all_actions.append(self.disable_collect_statistics_action)

        self.save_statistics_to_json_action = QAction("&Save statistics", self)
        self.save_statistics_to_json_action.setIcon(QIcon("aset_ui/resources/statistics_save.svg"))
        self.save_statistics_to_json_action.setStatusTip("Save the statistics to a .json file.")
        self.save_statistics_to_json_action.triggered.connect(self.save_statistics_to_json_task)
        self.save_statistics_to_json_action.setEnabled(False)
        self._all_actions.append(self.save_statistics_to_json_action)

        # set up the menu bar
        self.menubar = self.menuBar()
        self.menubar.setFont(MENU_FONT)

        self.file_menu = self.menubar.addMenu("&File")
        self.file_menu.setFont(MENU_FONT)
        self.file_menu.addAction(self.exit_action)

        self.document_base_menu = self.menubar.addMenu("&Document Base")
        self.document_base_menu.setFont(MENU_FONT)
        self.document_base_menu.addAction(self.show_document_base_creator_widget_action)
        self.document_base_menu.addSeparator()
        self.document_base_menu.addAction(self.load_document_base_from_bson_action)
        self.document_base_menu.addAction(self.save_document_base_to_bson_action)
        self.document_base_menu.addSeparator()
        self.document_base_menu.addAction(self.save_table_to_csv_action)
        self.document_base_menu.addSeparator()
        self.document_base_menu.addAction(self.add_attribute_action)
        self.document_base_menu.addAction(self.remove_attribute_action)
        self.document_base_menu.addSeparator()
        self.document_base_menu.addAction(self.forget_matches_for_attribute_action)
        self.document_base_menu.addAction(self.forget_matches_action)

        self.preprocessing_menu = self.menubar.addMenu("&Preprocessing")
        self.preprocessing_menu.setFont(MENU_FONT)
        self.preprocessing_menu.addAction(self.load_and_run_default_preprocessing_phase_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(self.load_preprocessing_phase_from_config_action)
        self.preprocessing_menu.addAction(self.save_preprocessing_phase_to_config_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(self.run_preprocessing_phase_action)

        self.matching_menu = self.menubar.addMenu("&Matching")
        self.matching_menu.setFont(MENU_FONT)
        self.matching_menu.addAction(self.load_and_run_default_matching_phase_action)
        self.matching_menu.addSeparator()
        self.matching_menu.addAction(self.load_matching_phase_from_config_action)
        self.matching_menu.addAction(self.save_matching_phase_to_config_action)
        self.matching_menu.addSeparator()
        self.matching_menu.addAction(self.run_matching_phase_action)

        self.statistics_menu = self.menubar.addMenu("&Statistics")
        self.statistics_menu.setFont(MENU_FONT)
        self.statistics_menu.addAction(self.enable_collect_statistics_action)
        self.statistics_menu.addAction(self.disable_collect_statistics_action)
        self.statistics_menu.addSeparator()
        self.statistics_menu.addAction(self.save_statistics_to_json_action)

        # start menu
        self.start_menu_widget = QWidget()
        self.start_menu_layout = QVBoxLayout(self.start_menu_widget)
        self.start_menu_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_layout.setSpacing(30)
        self.start_menu_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.start_menu_widget.setMaximumWidth(400)

        self.start_menu_header = QLabel("Welcome to ASET!")
        self.start_menu_header.setFont(HEADER_FONT)
        self.start_menu_layout.addWidget(self.start_menu_header)

        self.start_menu_create_new_document_base_widget = QWidget()
        self.start_menu_create_new_document_base_layout = QVBoxLayout(self.start_menu_create_new_document_base_widget)
        self.start_menu_create_new_document_base_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_create_new_document_base_layout.setSpacing(10)
        self.start_menu_layout.addWidget(self.start_menu_create_new_document_base_widget)

        self.start_menu_create_new_document_base_subheader = QLabel("Create a new document base.")
        self.start_menu_create_new_document_base_subheader.setFont(SUBHEADER_FONT)
        self.start_menu_create_new_document_base_layout.addWidget(self.start_menu_create_new_document_base_subheader)

        self.start_menu_create_new_document_base_wrapper_widget = QWidget()
        self.start_menu_create_new_document_base_wrapper_layout = QHBoxLayout(
            self.start_menu_create_new_document_base_wrapper_widget)
        self.start_menu_create_new_document_base_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_create_new_document_base_wrapper_layout.setSpacing(20)
        self.start_menu_create_new_document_base_wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.start_menu_create_new_document_base_layout.addWidget(
            self.start_menu_create_new_document_base_wrapper_widget)

        self.start_menu_create_document_base_button = QPushButton()
        self.start_menu_create_document_base_button.setFixedHeight(45)
        self.start_menu_create_document_base_button.setFixedWidth(45)
        self.start_menu_create_document_base_button.setIcon(QIcon("aset_ui/resources/two_documents.svg"))
        self.start_menu_create_document_base_button.clicked.connect(self.show_document_base_creator_widget_task)
        self.start_menu_create_new_document_base_wrapper_layout.addWidget(self.start_menu_create_document_base_button)

        self.start_menu_create_document_base_label = QLabel(
            "Create a new document base from a directory\nof .txt files and a list of attribute names.")
        self.start_menu_create_document_base_label.setFont(LABEL_FONT)
        self.start_menu_create_new_document_base_wrapper_layout.addWidget(self.start_menu_create_document_base_label)

        self.start_menu_load_document_base_widget = QWidget()
        self.start_menu_load_document_base_layout = QVBoxLayout(self.start_menu_load_document_base_widget)
        self.start_menu_load_document_base_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_load_document_base_layout.setSpacing(10)
        self.start_menu_layout.addWidget(self.start_menu_load_document_base_widget)

        self.start_menu_load_document_base_subheader = QLabel("Load an existing document base.")
        self.start_menu_load_document_base_subheader.setFont(SUBHEADER_FONT)
        self.start_menu_load_document_base_layout.addWidget(self.start_menu_load_document_base_subheader)

        self.start_menu_load_document_base_wrapper_widget = QWidget()
        self.start_menu_load_document_base_wrapper_layout = QHBoxLayout(
            self.start_menu_load_document_base_wrapper_widget)
        self.start_menu_load_document_base_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self.start_menu_load_document_base_wrapper_layout.setSpacing(20)
        self.start_menu_load_document_base_wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.start_menu_load_document_base_layout.addWidget(self.start_menu_load_document_base_wrapper_widget)

        self.start_menu_load_document_base_button = QPushButton()
        self.start_menu_load_document_base_button.setFixedHeight(45)
        self.start_menu_load_document_base_button.setFixedWidth(45)
        self.start_menu_load_document_base_button.setIcon(QIcon("aset_ui/resources/folder.svg"))
        self.start_menu_load_document_base_button.clicked.connect(self.load_document_base_from_bson_task)
        self.start_menu_load_document_base_wrapper_layout.addWidget(self.start_menu_load_document_base_button)

        self.start_menu_load_document_base_label = QLabel("Load an existing document base\nfrom a .bson file.")
        self.start_menu_load_document_base_label.setFont(LABEL_FONT)
        self.start_menu_load_document_base_wrapper_layout.addWidget(self.start_menu_load_document_base_label)

        # main UI
        self.central_widget = QWidget(self)
        self.central_widget_layout = QHBoxLayout(self.central_widget)
        self.central_widget_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.central_widget)

        self.document_base_creator_widget = DocumentBaseCreatorWidget(self)
        self.document_base_viewer_widget = DocumentBaseViewerWidget(self)
        self.interactive_matching_widget = InteractiveMatchingWidget(self)

        self.document_base_creator_widget.hide()
        self.document_base_viewer_widget.hide()
        self.interactive_matching_widget.hide()
        self.central_widget_layout.addWidget(self.start_menu_widget)
        self.central_widget_layout.update()

        self.resize(1400, 800)
        self.show()

        logger.info("Initialized MainWindow.")
