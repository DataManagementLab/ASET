import csv
import logging
import os
import pathlib

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QProgressBar, QTextEdit

from aset.embedding.aggregation import AttributeEmbeddingMethod
from aset.extraction.common import Extraction
from aset.extraction.extractionstage import ExtractionStage
from aset.matching import strategies
from aset.matching.common import Attribute
from aset.matching.matchingstage import MatchingStage
from aset.matching.strategies import DFSExploration
from aset_ui.onlinephase.attributes import AttributesInputWidget
from aset_ui.onlinephase.files import SourceFileWidget, TargetFileWidget
from aset_ui.util import HEADER_FONT, SUBHEADER_FONT, LABEL_FONT, LABEL_FONT_BOLD, \
    LABEL_CODE_FONT

logger = logging.getLogger(__name__)


class OnlinePhaseWorker(QObject):
    """Worker that executes the online phase."""

    finished: pyqtSignal = pyqtSignal()
    progress: pyqtSignal = pyqtSignal(float)
    next: pyqtSignal = pyqtSignal()
    matching_preparation_finished: pyqtSignal = pyqtSignal()
    feedback_request: pyqtSignal = pyqtSignal(str, str, str)

    def __init__(self, source_path: str, target_path: str, attributes: dict):
        """
        Initialize the online phase worker with paths to the preprocessed document collection file and the file where to
        save the resulting table.

        :param source_path: path to the preprocessed document collection file
        :param target_path: path to the file where the resulting table should be stored
        :param attributes: attributes of the resulting table
        """
        super(OnlinePhaseWorker, self).__init__()
        self.source_path: str = source_path
        self.target_path: str = target_path
        self.extraction_stage: ExtractionStage or None = None
        self.matching_stage: MatchingStage or None = None
        self.attributes: dict = attributes

    def run(self):
        """Main code of the online phase."""

        # load the extraction stage
        logger.info(f"Load extraction stage from '{self.source_path}'.")
        self.next.emit()
        self.progress.emit(-1.)

        with open(self.source_path, encoding="utf-8") as file:
            self.extraction_stage = ExtractionStage.from_json_str(file.read())

        self.next.emit()

        # load the matching stage
        logger.info(f"Load the matching stage.")
        self.progress.emit(-1.)

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
        self.next.emit()

        # compute attribute embeddings
        logger.info("Compute attribute embeddings.")
        self.progress.emit(-1.)
        self.matching_stage.compute_attribute_embeddings()
        self.next.emit()

        # incorporate example values
        logger.info("Incorporate example values.")
        self.progress.emit(-1.)
        self.matching_stage.incorporate_example_mentions(mentions)
        self.next.emit()

        # interactive matching
        logger.info("Interactive matching.")
        self.progress.emit(-1.)

        def query_user(document_index: int, attribute: Attribute, extraction: Extraction, num_user_queries: int):
            print("{:3.3} '{}'?  '{}'  ==>  ?".format(
                str(num_user_queries) + ".",
                attribute.label,
                extraction.mention
            ))
            self.feedback_request.emit(attribute.label, extraction.mention, extraction.context)
            return True if input("y/n: ") == "y" else False

        strategies.query_user = query_user

        self.matching_preparation_finished.emit()
        self.matching_stage.match_extractions_to_attributes()

        self.next.emit()

        # save resulting table
        logger.info("Save resulting table.")
        self.progress.emit(-1.)
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
        self.next.emit()

        self.progress.emit(1.)

        logger.info("All done.")
        self.finished.emit()

    def incorporate_feedback(self, feedback: bool):
        print("Feedback would now be incorporated.")


class MatchingPreparationWidget(QWidget):
    """Widget to start and monitor the preparation of the matching process."""

    def __init__(self, parent):
        super(MatchingPreparationWidget, self).__init__()
        self.parent = parent
        self.current_step = -1

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 10)

        self.subheader = QLabel("4. Start the matching process.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.steps_widget = QWidget()
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setContentsMargins(30, 5, 0, 10)

        self.steps = [
            QLabel("- Load preprocessed document collection."),
            QLabel("- Load matching stage."),
            QLabel("- Compute attribute embeddings."),
            QLabel("- Incorporate example values."),
            QLabel("- Interactive matching."),
            QLabel("- Store resulting table.")
        ]
        for label in self.steps:
            label.setFont(LABEL_FONT)
            self.steps_layout.addWidget(label)

        self.steps_widget.setLayout(self.steps_layout)
        self.layout.addWidget(self.steps_widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.start_button = QPushButton()
        self.start_button.setText("Start Matching")
        self.start_button.clicked.connect(self.parent.start_matching_preparation)
        self.layout.addWidget(self.start_button)

        self.setLayout(self.layout)

    def next(self):
        if self.current_step >= 0:
            self.steps[self.current_step].setFont(LABEL_FONT)
        else:
            self.steps[-1].setFont(LABEL_FONT)

        self.current_step += 1

        if self.current_step < len(self.steps):
            self.steps[self.current_step].setFont(LABEL_FONT_BOLD)
        else:
            self.current_step = -1

    def progress(self, fraction_done: float):
        if fraction_done < 0:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(int(fraction_done * 100))


class InteractiveMatchingWindow(QWidget):
    """Window for the interactive matching."""

    def __init__(self, parent):
        super(InteractiveMatchingWindow, self).__init__()
        self.parent = parent

        self.setWindowTitle("ASET: Interactive Matching")

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 15, 20, 15)

        self.header = QLabel("ASET: Interactive Matching")
        self.header.setFont(HEADER_FONT)
        self.layout.addWidget(self.header)

        self.description = QLabel("You may give feedback on ASET's matching decisions to improve the resulting table.")
        self.description.setFont(LABEL_FONT)
        self.layout.addWidget(self.description)

        self.matching_widget = QWidget()
        self.matching_layout = QVBoxLayout()
        self.matching_layout.setContentsMargins(0, 10, 0, 10)

        self.subheader = QLabel("Matching attribute: ")
        self.subheader.setFont(SUBHEADER_FONT)
        self.matching_layout.addWidget(self.subheader)

        self.mention = QTextEdit()
        self.mention.setReadOnly(True)
        self.mention.setFont(LABEL_CODE_FONT)
        self.mention.textCursor().insertHtml("")
        self.matching_layout.addWidget(self.mention)

        self.matching_widget.setLayout(self.matching_layout)
        self.layout.addWidget(self.matching_widget)

        self.label = QLabel("Does the highlighted mention match the attribute?")
        self.label.setFont(LABEL_FONT)
        self.layout.addWidget(self.label)

        self.buttons_widget = QWidget(self)
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.yes_button = QPushButton("Yes")
        self.yes_button.clicked.connect(self.yes_button_clicked)
        self.buttons_layout.addWidget(self.yes_button)

        self.no_button = QPushButton("No")
        self.no_button.clicked.connect(self.no_button_clicked)
        self.buttons_layout.addWidget(self.no_button)

        self.buttons_widget.setLayout(self.buttons_layout)
        self.layout.addWidget(self.buttons_widget)

        self.setLayout(self.layout)
        self.layout.addStretch()

        # lock the window size
        self.setFixedSize(800, 400)

        logger.debug("Initialized interactive matching window.")

    def yes_button_clicked(self, _):
        self.parent.incorporate_feedback(True)

    def no_button_clicked(self, _):
        self.parent.incorporate_feedback(False)

    def closeEvent(self, _):
        logger.info("Close interactive matching window.")
        self.parent.show()
        self.parent.matching_finished()


class OnlinePhaseWindow(QWidget):
    """Window of the online phase."""

    feedback_response: pyqtSignal = pyqtSignal(bool)

    def __init__(self, parent):
        super(OnlinePhaseWindow, self).__init__()
        self.parent = parent

        self.interactive_matching_window = None

        self.online_phase_worker = None
        self.worker_thread = None

        # layout, title, and header
        self.setWindowTitle("ASET: Online Matching Phase")
        self.resize(1000, 1000)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 15, 20, 15)

        self.header = QLabel("ASET: Online Matching Phase")
        self.header.setFont(HEADER_FONT)
        self.layout.addWidget(self.header)

        # widgets
        self.source_file_widget = SourceFileWidget(self)
        self.layout.addWidget(self.source_file_widget)

        self.target_file_widget = TargetFileWidget(self)
        self.layout.addWidget(self.target_file_widget)

        self.attributes_widget = AttributesInputWidget(self)
        self.layout.addWidget(self.attributes_widget)
        self.layout.setStretch(3, 1)

        self.matching_preparation_widget = MatchingPreparationWidget(self)
        self.layout.addWidget(self.matching_preparation_widget)

        # select directory
        self.setLayout(self.layout)
        self.layout.addStretch()

        logger.debug("Initialized online phase window.")

    def check_source_file(self):
        """Check that the preprocessed document collection file is valid."""
        file_path = self.source_file_widget.get_filepath()
        logger.info(f"Check source file path '{file_path}'.")

        # check that the path leads to a file
        if not os.path.isfile(file_path):
            logger.error("The provided source file path is invalid!")
            self.source_file_widget.give_feedback("The provided source file path is invalid!")
            return False

        return True

    def check_target_file(self):
        """Check that the target file path is valid."""
        file_path = self.target_file_widget.get_filepath()
        logger.info(f"Check target file path '{file_path}'.")

        # check that the path without the final part leads to a folder
        if not os.path.isdir("/".join(pathlib.Path(file_path).parts[:-1])):
            logger.error("The provided target file path is invalid!")
            self.target_file_widget.give_feedback("The provided target file path is invalid!")
            return False

        # check that the entire path is not a folder
        if os.path.isdir(file_path):
            logger.error("The provided target file path is invalid!")
            self.target_file_widget.give_feedback("The provided target file path is invalid!")
            return False

        if os.path.isfile(file_path):
            logger.info("Resulting table will overwrite existing file.")
            self.target_file_widget.give_feedback("Existing file will be overwritten.")

        return True

    def check_attributes(self):
        """Check that the attributes are valid."""
        attributes = set()

        for attribute_widget in self.attributes_widget.get_attribute_widgets():
            attribute_name = attribute_widget.name_edit.text()

            if attribute_name.strip() == "":
                logger.error("Attribute name cannot be an empty string!")
                self.attributes_widget.give_feedback("Attribute names cannot be empty strings!")
                return False

            if attribute_name in attributes:
                logger.error("Attribute name already exists!")
                self.attributes_widget.give_feedback("Duplicate attribute names are not allowed!")
                return False

            for example_widget in attribute_widget.example_widgets:
                example_value = example_widget.value_edit.text()

                if example_value.strip() == "":
                    logger.error("Empty example values are not allowed!")
                    self.attributes_widget.give_feedback("Empty example values are not allowed!")
                    return False

            attributes.add(attribute_name)

        if len(attributes) == 0:
            logger.error("There must be at least one attribute!")
            self.attributes_widget.give_feedback("There must be at least one attribute!")
            return False

        return True

    def matching_preparation_finished(self):
        # update the GUI
        self.matching_preparation_widget.progress_bar.setMinimum(0)
        self.matching_preparation_widget.progress_bar.setMaximum(100)
        self.matching_preparation_widget.progress_bar.setValue(100)

        # go into interactive matching
        logger.info("Start interactive matching.")
        self.interactive_matching_window = InteractiveMatchingWindow(self)
        self.hide()
        self.interactive_matching_window.show()

    def matching_finished(self):
        self.interactive_matching_window.hide()
        self.show()

        # update the GUI
        self.source_file_widget.setEnabled(True)
        self.target_file_widget.setEnabled(True)
        self.attributes_widget.setEnabled(True)
        self.matching_preparation_widget.start_button.setEnabled(True)
        for step in self.matching_preparation_widget.steps:
            step.setFont(LABEL_FONT)
        self.matching_preparation_widget.current_step = -1

    def feedback_request(self, label, mention, context):
        self.interactive_matching_window.subheader.setText("Matching attribute: " + label)
        formatted_text = context[:context.index(
            mention)] + "<span style='background-color: #FFFF00'><b>" + mention + "</b></span>" + context[context.index(
            mention) + len(mention):]
        self.interactive_matching_window.mention.setText("")
        self.interactive_matching_window.mention.textCursor().insertHtml(formatted_text)

        self.interactive_matching_window.buttons_widget.setEnabled(True)

    def incorporate_feedback(self, value):
        self.interactive_matching_window.buttons_widget.setEnabled(False)
        self.feedback_response.emit(value)

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
        self.source_file_widget.setEnabled(False)
        self.target_file_widget.setEnabled(False)
        self.attributes_widget.setEnabled(False)
        self.matching_preparation_widget.start_button.setEnabled(False)

        # gather the attributes
        attributes = {}
        for attribute_widget in self.attributes_widget.get_attribute_widgets():
            attribute_name = attribute_widget.name_edit.text()
            example_values = []
            for example_widget in attribute_widget.example_widgets:
                example_values.append(example_widget.value_edit.text())
            attributes[attribute_name] = example_values

        # start matching preparation
        self.online_phase_worker = OnlinePhaseWorker(
            source_path=self.source_file_widget.get_filepath(),
            target_path=self.target_file_widget.get_filepath(),
            attributes=attributes
        )

        self.worker_thread = QThread()
        self.online_phase_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.online_phase_worker.run)
        self.online_phase_worker.finished.connect(self.matching_finished)
        self.online_phase_worker.finished.connect(self.worker_thread.quit)
        self.online_phase_worker.finished.connect(self.online_phase_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.online_phase_worker.next.connect(self.matching_preparation_widget.next)
        self.online_phase_worker.progress.connect(self.matching_preparation_widget.progress)
        self.online_phase_worker.matching_preparation_finished.connect(self.matching_preparation_finished)
        self.online_phase_worker.feedback_request.connect(self.feedback_request)
        self.feedback_response.connect(self.online_phase_worker.incorporate_feedback)

        self.worker_thread.start()

    def closeEvent(self, event):
        """When window closed, go back to parent."""
        logger.info("Close online phase window.")
        self.parent.show()
