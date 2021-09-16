import csv
import logging
import os
import pathlib

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QMutex, QWaitCondition
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, \
    QScrollArea, QFrame, QProgressBar, QTextEdit

from aset.embedding.aggregation import AttributeEmbeddingMethod
from aset.extraction.common import Extraction
from aset.extraction.extractionstage import ExtractionStage
from aset.matching import strategies
from aset.matching.common import Attribute
from aset.matching.matchingstage import MatchingStage
from aset.matching.strategies import DFSExploration
from aset_ui.util import SUBHEADER_FONT, LABEL_FONT, LABEL_FONT_ITALIC, LABEL_FONT_BOLD, HEADER_FONT, LABEL_CODE_FONT

logger = logging.getLogger(__name__)


########################################################################################################################
# source and target file selection
########################################################################################################################
class FileSelectorWidget(QWidget):
    """Widget to select the source/target file."""

    def __init__(self, parent, subheader_text, label_text, placeholder_text, dialog_text):
        super(FileSelectorWidget, self).__init__(parent)
        self._parent = parent
        self._dialog_text = dialog_text

        # layout, header, and label
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 10, 0, 10)

        self._subheader = QLabel(subheader_text)
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._label = QLabel(label_text)
        self._label.setFont(LABEL_FONT)
        self._label.setWordWrap(True)
        self._layout.addWidget(self._label)

        # file path edit and button
        self._file_path_widget = QWidget()
        self._file_path_layout = QHBoxLayout()
        self._file_path_widget.setLayout(self._file_path_layout)
        self._file_path_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._file_path_widget)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder_text)
        self._edit.textChanged.connect(self._filepath_changed)
        self._file_path_layout.addWidget(self._edit)

        self._button = QPushButton("Select File")
        self._button.clicked.connect(self._button_pressed)
        self._file_path_layout.addWidget(self._button)

        # feedback label
        self._feedback_label = QLabel(" ")
        self._feedback_label.setFont(LABEL_FONT_ITALIC)
        self._layout.addWidget(self._feedback_label)

    def _button_pressed(self):
        path = str(QFileDialog.getOpenFileName(self, self._dialog_text)[0])
        if path != "":
            self._edit.setText(path)
            self._parent.check_source_file()

    def _filepath_changed(self):
        self._feedback_label.setStyleSheet("color: black")
        self._feedback_label.setText(" ")

    def give_feedback(self, feedback):
        self._feedback_label.setStyleSheet("color: red")
        self._feedback_label.setText(feedback)

    def get_filepath(self):
        return self._edit.text()


########################################################################################################################
# attributes
########################################################################################################################
class AttributesInputWidget(QWidget):
    """Widget to enter the attributes and provide examples for them."""

    def __init__(self, parent):
        super(AttributesInputWidget, self).__init__(parent)
        self._parent = parent

        # layout and header
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(self._layout)

        self._subheader = QLabel("3. Enter attributes to extract from the documents.")
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._label = QLabel("Each attribute must have a unique name. You may also provide example values for each "
                             "attribute.")
        self._label.setWordWrap(True)
        self._label.setFont(LABEL_FONT)
        self._layout.addWidget(self._label)

        # list of attributes
        self._attribute_widgets = []

        self._list_widget = QWidget()
        self._list_layout = QHBoxLayout()
        self._list_widget.setLayout(self._list_layout)
        self._list_layout.setContentsMargins(10, 10, 10, 10)
        self._list_layout.setAlignment(Qt.AlignTop)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._list_widget)
        self._layout.addWidget(self._scroll_area)

        # add attribute button
        self._add_attribute_button = QPushButton("Add Attribute")
        self._add_attribute_button.clicked.connect(self.add_attribute)
        self._add_attribute_button.setFixedWidth(100)
        self._list_layout.addWidget(self._add_attribute_button)

        # feedback label
        self._feedback_label = QLabel(" ")
        self._feedback_label.setFont(LABEL_FONT_ITALIC)
        self._layout.addWidget(self._feedback_label)

    def add_attribute(self):
        new_attribute_widget = AttributeWidget(self)
        self._attribute_widgets.append(new_attribute_widget)
        self._list_layout.addWidget(new_attribute_widget)
        self._list_layout.removeWidget(self._add_attribute_button)
        self._list_layout.addWidget(self._add_attribute_button)

        self.attributes_changed()

    def remove_attribute(self, attribute_widget):
        attribute_widget.hide()
        self._list_layout.removeWidget(attribute_widget)
        self._attribute_widgets.remove(attribute_widget)
        attribute_widget.deleteLater()

        self.attributes_changed()

    def attributes_changed(self):
        self._scroll_area.update()

        self._feedback_label.setStyleSheet("color: black")
        self._feedback_label.setText(" ")

    def give_feedback(self, feedback):
        self._feedback_label.setStyleSheet("color: red")
        self._feedback_label.setText(feedback)

    def get_attribute_widgets(self):
        return self._attribute_widgets


class AttributeWidget(QFrame):
    """Widget to display a single attribute."""

    def __init__(self, parent):
        super(AttributeWidget, self).__init__(parent)
        self._parent = parent

        # layout
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setFixedWidth(300)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setAlignment(Qt.AlignTop)

        # top widget: attribute name and remove attribute button
        self._top_widget = QWidget()
        self._top_layout = QHBoxLayout()
        self._top_widget.setLayout(self._top_layout)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._top_widget)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("attribute name")
        self._name_edit.textChanged.connect(self._parent.attributes_changed)
        self._top_layout.addWidget(self._name_edit)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(lambda: self._parent.remove_attribute(self))
        self._top_layout.addWidget(self._remove_button)

        # list of examples
        self._example_widgets = []

        self._examples_widget = QWidget()
        self._examples_layout = QVBoxLayout()
        self._examples_widget.setLayout(self._examples_layout)
        self._examples_layout.setContentsMargins(30, 0, 0, 0)
        self._layout.addWidget(self._examples_widget)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout()
        self._list_widget.setLayout(self._list_layout)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._examples_layout.addWidget(self._list_widget)

        self._add_example_button = QPushButton("Add Example Value")
        self._add_example_button.clicked.connect(self.add_example)
        self._examples_layout.addWidget(self._add_example_button)

    def get_name(self):
        return self._name_edit.text()

    def get_example_widgets(self):
        return self._example_widgets

    def add_example(self):
        new_example_widget = ExampleWidget(self)
        self._example_widgets.append(new_example_widget)
        self._list_layout.addWidget(new_example_widget)

        self.examples_changed()

    def remove_example(self, example_widget):
        example_widget.hide()
        self._list_layout.removeWidget(example_widget)
        self._example_widgets.remove(example_widget)
        example_widget.deleteLater()

        self.examples_changed()

    def examples_changed(self):
        self._parent.attributes_changed()


class ExampleWidget(QWidget):
    """Widget to display a single example."""

    def __init__(self, parent):
        super(ExampleWidget, self).__init__(parent)
        self._parent = parent

        # layout
        self._layout = QHBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # value and remove button
        self._value_edit = QLineEdit()
        self._value_edit.setPlaceholderText("example value")
        self._value_edit.textChanged.connect(self._parent.examples_changed)
        self._layout.addWidget(self._value_edit)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(lambda: self._parent.remove_example(self))
        self._layout.addWidget(self._remove_button)

    def get_value(self):
        return self._value_edit.text()


########################################################################################################################
# matching status
########################################################################################################################
class MatchingStatusWidget(QWidget):
    """Widget to start and monitor the preparation of the matching process."""

    def __init__(self, parent):
        super(MatchingStatusWidget, self).__init__()
        self._parent = parent
        self._current_step = -1

        # set up the widgets
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(self._layout)

        self._subheader = QLabel("4. Start the matching process.")
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._steps_widget = QWidget()
        self._steps_layout = QVBoxLayout()
        self._steps_layout.setContentsMargins(30, 5, 0, 10)
        self._steps_widget.setLayout(self._steps_layout)
        self._layout.addWidget(self._steps_widget)

        self._steps = [
            QLabel("- Load preprocessed document collection."),
            QLabel("- Load matching stage."),
            QLabel("- Compute attribute embeddings."),
            QLabel("- Incorporate example values."),
            QLabel("- Interactive matching."),
            QLabel("- Store resulting table.")
        ]
        for label in self._steps:
            label.setFont(LABEL_FONT)
            self._steps_layout.addWidget(label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._layout.addWidget(self._progress_bar)

        self._start_button = QPushButton()
        self._start_button.setText("Start Matching")
        self._start_button.clicked.connect(self._parent.start_matching_preparation)
        self._layout.addWidget(self._start_button)

    def next(self):
        if self._current_step >= 0:
            self._steps[self._current_step].setFont(LABEL_FONT)
        else:
            self._steps[-1].setFont(LABEL_FONT)

        self._current_step += 1

        if self._current_step < len(self._steps):
            self._steps[self._current_step].setFont(LABEL_FONT_BOLD)
        else:
            self._current_step = -1

    def reset(self):
        self._current_step = -1
        self._start_button.setEnabled(True)
        for step in self._steps:
            step.setFont(LABEL_FONT)

    def disable_start_button(self):
        self._start_button.setEnabled(False)

    def progress(self, fraction_done: float):
        if fraction_done < 0:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(0)
        else:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(int(fraction_done * 100))


########################################################################################################################
# interactive matching
########################################################################################################################
class InteractiveMatchingWindow(QWidget):
    """Window for the interactive matching."""

    def __init__(self, parent):
        super(InteractiveMatchingWindow, self).__init__()
        self._parent = parent

        self.setWindowTitle("ASET: Interactive Matching")

        # set up the widgets
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(20, 15, 20, 15)
        self.setLayout(self._layout)

        self._header = QLabel("ASET: Interactive Matching")
        self._header.setFont(HEADER_FONT)
        self._layout.addWidget(self._header)

        self._description = QLabel("You may give feedback on ASET's matching decisions to improve the resulting table.")
        self._description.setFont(LABEL_FONT)
        self._layout.addWidget(self._description)

        self._matching_widget = QWidget()
        self._matching_layout = QVBoxLayout()
        self._matching_layout.setContentsMargins(0, 10, 0, 10)
        self._matching_widget.setLayout(self._matching_layout)

        self._subheader = QLabel("Matching attribute: ")
        self._subheader.setFont(SUBHEADER_FONT)
        self._matching_layout.addWidget(self._subheader)

        self._mention = QTextEdit()
        self._mention.setReadOnly(True)
        self._mention.setFont(LABEL_CODE_FONT)
        self._mention.textCursor().insertHtml("")
        self._matching_layout.addWidget(self._mention)

        self._layout.addWidget(self._matching_widget)

        self._label = QLabel("Does the highlighted mention match the attribute?")
        self._label.setFont(LABEL_FONT)
        self._layout.addWidget(self._label)

        self._buttons_widget = QWidget(self)
        self._buttons_layout = QHBoxLayout()
        self._buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._buttons_widget.setLayout(self._buttons_layout)

        self._yes_button = QPushButton("Yes")
        self._yes_button.clicked.connect(self._yes_button_clicked)
        self._buttons_layout.addWidget(self._yes_button)

        self._no_button = QPushButton("No")
        self._no_button.clicked.connect(self._no_button_clicked)
        self._buttons_layout.addWidget(self._no_button)

        self._layout.addWidget(self._buttons_widget)

        # lock the window size
        self.setFixedSize(800, 400)

        logger.debug("Initialized interactive matching window.")

    def _yes_button_clicked(self, _):
        self._parent.incorporate_feedback(True)

    def _no_button_clicked(self, _):
        self._parent.incorporate_feedback(False)

    def feedback_request(self, label, mention, context):
        self._subheader.setText("Matching attribute: " + label)
        formatted_text = context[:context.index(mention)] + \
                         "<span style='background-color: #FFFF00'><b>" + mention + "</b></span>" + \
                         context[context.index(mention) + len(mention):]
        self._mention.setText("")
        self._mention.textCursor().insertHtml(formatted_text)

        self._buttons_widget.setEnabled(True)

    def disable_buttons(self):
        self._buttons_widget.setEnabled(False)

    def closeEvent(self, _):
        logger.info("Close interactive matching window.")
        self.hide()
        self._parent.show()
        self._parent.matching_finished()
        self.deleteLater()


########################################################################################################################
# worker thread
########################################################################################################################
class OnlinePhaseWorker(QObject):
    """Worker that executes the online phase."""

    matching_finished_to_ui: pyqtSignal = pyqtSignal()
    progress_to_ui: pyqtSignal = pyqtSignal(float)
    next_to_ui: pyqtSignal = pyqtSignal()
    matching_preparation_finished_to_ui: pyqtSignal = pyqtSignal()
    feedback_request_to_ui: pyqtSignal = pyqtSignal(str, str, str)

    def __init__(self, source_path, target_path, attributes, mutex, cond):
        """
        Initialize the online phase worker with paths to the preprocessed document collection file and the file where to
        save the resulting table.
        """
        super(OnlinePhaseWorker, self).__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.extraction_stage = None
        self.matching_stage = None
        self.attributes = attributes

        self.mutex = mutex
        self.cond = cond
        self.value = False

    def run(self):
        """Main code of the online phase."""

        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # load the extraction stage
        logger.info(f"Load extraction stage from '{self.source_path}'.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)
        with open(self.source_path, encoding="utf-8") as file:
            self.extraction_stage = ExtractionStage.from_json_str(file.read())
        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # load the matching stage
        logger.info("Load the matching stage.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)

        attributes = []
        mentions = []
        for attribute_name, attribute_mentions in self.attributes.items():
            attributes.append(Attribute(attribute_name))
            mentions.append(attribute_mentions)

        self.matching_stage = MatchingStage(
            documents=self.extraction_stage.documents,
            attributes=attributes,
            strategy=DFSExploration(
                max_children=2,
                explore_far_factor=1.15,
                max_distance=0.3,
                max_interactions=24
            ),
            embedding_method=AttributeEmbeddingMethod(),
        )
        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # compute attribute embeddings
        logger.info("Compute attribute embeddings.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)
        self.matching_stage.compute_attribute_embeddings()
        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # incorporate example values
        logger.info("Incorporate example values.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)
        self.matching_stage.incorporate_example_mentions(mentions)
        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # interactive matching
        logger.info("Interactive matching.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)

        def query_user(document_index: int, attribute: Attribute, extraction: Extraction, num_user_queries: int):
            # noinspection PyUnresolvedReferences
            self.feedback_request_to_ui.emit(attribute.label, extraction.mention, extraction.context)

            self.mutex.lock()
            try:
                self.cond.wait(self.mutex)
            finally:
                self.mutex.unlock()
            return self.value

        strategies.query_user = query_user

        # noinspection PyUnresolvedReferences
        self.matching_preparation_finished_to_ui.emit()
        self.matching_stage.match_extractions_to_attributes()

        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()

        # save resulting table
        logger.info("Save resulting table.")
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(-1.)
        with open(self.target_path, "w", newline="", encoding="utf-8") as file:
            fieldnames = [attribute.label for attribute in self.matching_stage.attributes]
            file.write(",".join(fieldnames) + "\n")
            writer = csv.DictWriter(file, fieldnames=fieldnames, dialect="excel")
            for row in self.matching_stage.rows:
                row_dict = {}
                for attribute_name, extraction in row.extractions.items():
                    if extraction is None or extraction.value is None:  # no match has been found or no value
                        row_dict[attribute_name] = "-"
                    else:
                        row_dict[attribute_name] = extraction.value
                writer.writerow(row_dict)
        # noinspection PyUnresolvedReferences
        self.next_to_ui.emit()
        # noinspection PyUnresolvedReferences
        self.progress_to_ui.emit(1.)

        logger.info("All done.")
        # noinspection PyUnresolvedReferences
        self.matching_finished_to_ui.emit()


########################################################################################################################
# online phase window
########################################################################################################################
class OnlinePhaseWindow(QWidget):
    """Window of the online phase."""

    def __init__(self, parent):
        super(OnlinePhaseWindow, self).__init__()
        self._parent = parent

        self._interactive_matching_window = None

        self._online_phase_worker = None
        self._worker_thread = None
        self._mutex = QMutex()
        self._cond = QWaitCondition()

        # layout, title, and header
        self.setWindowTitle("ASET: Online Matching Phase")
        self.resize(1000, 1000)

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(20, 15, 20, 15)
        self.setLayout(self._layout)

        self._header = QLabel("ASET: Online Matching Phase")
        self._header.setFont(HEADER_FONT)
        self._layout.addWidget(self._header)

        # widgets
        self._source_file_widget = FileSelectorWidget(
            self,
            "1. Open a preprocessed document collection.",
            "ASET stores the preprocessed document collection as a .json file.",
            "source file path",
            "Select preprocessed document collection file"
        )
        self._layout.addWidget(self._source_file_widget)

        self._target_file_widget = FileSelectorWidget(
            self,
            "2. Choose where to store the resulting table.",
            "ASET stores the resulting table as a .csv file.",
            "target file path",
            "Select where to save the resulting table."
        )
        self._layout.addWidget(self._target_file_widget)

        self._attributes_widget = AttributesInputWidget(self)
        self._layout.addWidget(self._attributes_widget)
        self._layout.setStretch(3, 1)

        self._matching_status_widget = MatchingStatusWidget(self)
        self._layout.addWidget(self._matching_status_widget)

        self._layout.addStretch()

        logger.debug("Initialized online phase window.")

    def check_source_file(self):
        """Check that the preprocessed document collection file is valid."""
        file_path = self._source_file_widget.get_filepath()
        logger.info(f"Check source file path '{file_path}'.")

        # check that the path leads to a file
        if not os.path.isfile(file_path):
            logger.error("The provided source file path is invalid!")
            self._source_file_widget.give_feedback("The provided source file path is invalid!")
            return False

        return True

    def check_target_file(self):
        """Check that the target file path is valid."""
        file_path = self._target_file_widget.get_filepath()
        logger.info(f"Check target file path '{file_path}'.")

        # check that the path without the final part leads to a folder
        if not os.path.isdir("/".join(pathlib.Path(file_path).parts[:-1])):
            logger.error("The provided target file path is invalid!")
            self._target_file_widget.give_feedback("The provided target file path is invalid!")
            return False

        # check that the entire path is not a folder
        if os.path.isdir(file_path):
            logger.error("The provided target file path is invalid!")
            self._target_file_widget.give_feedback("The provided target file path is invalid!")
            return False

        if os.path.isfile(file_path):
            logger.info("Resulting table will overwrite existing file.")
            self._target_file_widget.give_feedback("Existing file will be overwritten.")

        return True

    def check_attributes(self):
        """Check that the attributes are valid."""
        attributes = set()

        for attribute_widget in self._attributes_widget.get_attribute_widgets():
            attribute_name = attribute_widget.get_name()

            if attribute_name.strip() == "":
                logger.error("Attribute name cannot be an empty string!")
                self._attributes_widget.give_feedback("Attribute names cannot be empty strings!")
                return False

            if attribute_name in attributes:
                logger.error("Attribute name already exists!")
                self._attributes_widget.give_feedback("Duplicate attribute names are not allowed!")
                return False

            for example_widget in attribute_widget.get_example_widgets():
                example_value = example_widget.get_value()

                if example_value.strip() == "":
                    logger.error("Empty example values are not allowed!")
                    self._attributes_widget.give_feedback("Empty example values are not allowed!")
                    return False

            attributes.add(attribute_name)

        if len(attributes) == 0:
            logger.error("There must be at least one attribute!")
            self._attributes_widget.give_feedback("There must be at least one attribute!")
            return False

        return True

    def matching_preparation_finished(self):
        # update the GUI
        self._matching_status_widget.progress(1)

        # go into interactive matching
        logger.info("Start interactive matching.")
        self.hide()
        self._interactive_matching_window.show()

    def matching_finished(self):

        self._interactive_matching_window.hide()
        self._interactive_matching_window.deleteLater()
        self.show()

        # update the GUI
        self._source_file_widget.setEnabled(True)
        self._target_file_widget.setEnabled(True)
        self._attributes_widget.setEnabled(True)
        self._matching_status_widget.reset()

    def incorporate_feedback(self, value):
        self._interactive_matching_window.disable_buttons()
        self._online_phase_worker.value = value
        self._cond.wakeAll()

    def start_matching_preparation(self, _):
        logger.info("Start matching preparations.")

        # check that the source file is valid
        if not self.check_source_file():
            return

        # check that the target file is valid
        if not self.check_target_file():
            return

        # check the attributes
        if not self.check_attributes():
            return

        # update the UI
        self._source_file_widget.setEnabled(False)
        self._target_file_widget.setEnabled(False)
        self._attributes_widget.setEnabled(False)
        self._matching_status_widget.disable_start_button()
        self._interactive_matching_window = InteractiveMatchingWindow(self)

        # gather the attributes
        attributes = {}
        for attribute_widget in self._attributes_widget.get_attribute_widgets():
            attribute_name = attribute_widget.get_name()
            example_values = []
            for example_widget in attribute_widget.get_example_widgets():
                example_values.append(example_widget.get_value())
            attributes[attribute_name] = example_values

        # start matching preparation
        self._online_phase_worker = OnlinePhaseWorker(
            source_path=self._source_file_widget.get_filepath(),
            target_path=self._target_file_widget.get_filepath(),
            attributes=attributes,
            mutex=self._mutex,
            cond=self._cond
        )

        self._worker_thread = QThread()
        self._online_phase_worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._online_phase_worker.run)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.matching_finished_to_ui.connect(self.matching_finished)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.matching_finished_to_ui.connect(self._worker_thread.quit)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.matching_finished_to_ui.connect(self._online_phase_worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        # noinspection PyUnresolvedReferences
        self._online_phase_worker.next_to_ui.connect(self._matching_status_widget.next)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.progress_to_ui.connect(self._matching_status_widget.progress)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.matching_preparation_finished_to_ui.connect(self.matching_preparation_finished)
        # noinspection PyUnresolvedReferences
        self._online_phase_worker.feedback_request_to_ui.connect(self._interactive_matching_window.feedback_request)

        self._worker_thread.start()

    def closeEvent(self, event):
        """When window closed, go back to parent."""
        logger.info("Close online phase window.")
        self._parent.show()
        self.deleteLater()
