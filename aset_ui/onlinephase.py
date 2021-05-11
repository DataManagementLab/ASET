"""Online Phase: Match between document collection and query."""
import logging
import os
import pathlib

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QFrame, \
    QProgressBar

from aset.embedding.aggregation import AttributeEmbeddingMethod
from aset.extraction.extractionstage import ExtractionStage
from aset.matching.common import Attribute
from aset.matching.matchingstage import MatchingStage
from aset.matching.strategies import DFSExploration
from aset_ui.util import HEADER_FONT, SUBHEADER_FONT, LABEL_FONT, LABEL_FONT_ITALIC, LABEL_FONT_BOLD

logger = logging.getLogger(__name__)


class OnlinePhaseWorker(QObject):
    """Worker that executes the online phase."""

    finished: pyqtSignal = pyqtSignal()
    progress: pyqtSignal = pyqtSignal(float)
    next: pyqtSignal = pyqtSignal()

    def __init__(self, source_path: str, target_path: str, attributes: dict):
        """
        Initialize the online phase worker with paths to the preprocessed document collection file and the file where to
        save the resulting table.

        :param source_path: path to the preprocessed document collection file
        :param target_path: path to the file where the resulting table should be stored
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
                max_interactions=25
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

        # start matching
        print("HERE WOULD BE THE MATCHING!")

        self.next.emit()

        logger.info("All done.")
        self.finished.emit()


class SourceFileWidget(QWidget):
    """Widget to select the source file."""

    def __init__(self, parent):
        super(SourceFileWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 0)

        self.subheader = QLabel("1. Open a preprocessed document collection.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.label = QLabel("ASET stores the preprocessed document collection as a .json file.")
        self.label.setFont(LABEL_FONT)
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select File")
        self.edit.setText(r"C:/Users/micha/Code/ASET/datasets/aviation/out.json")
        self.edit.textChanged.connect(self.parent.source_file_edit_changed)
        self.hbox_layout.addWidget(self.edit)

        self.button = QPushButton("Select File")
        self.button.clicked.connect(self.button_pressed)
        self.hbox_layout.addWidget(self.button)

        self.hbox_widget.setLayout(self.hbox_layout)
        self.layout.addWidget(self.hbox_widget)

        self.feedback_label = QLabel(" ")
        self.feedback_label.setFont(LABEL_FONT_ITALIC)
        self.feedback_label.setStyleSheet("color: red")
        self.layout.addWidget(self.feedback_label)

        self.setLayout(self.layout)

    def button_pressed(self, _):
        logger.info("Select preprocessed document collection file.")
        path = str(QFileDialog.getOpenFileName(self, "Select preprocessed document collection file")[0])
        if path != "":
            self.edit.setText(path)
            self.parent.check_source_file()


class TargetFileWidget(QWidget):
    """Widget to select the target file."""

    def __init__(self, parent):
        super(TargetFileWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 0)

        self.subheader = QLabel("2. Choose where to store the resulting table.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.label = QLabel("ASET stores the resulting table as a .csv file.")
        self.label.setFont(LABEL_FONT)
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select File")
        self.edit.setText(r"C:/Users/micha/Code/ASET/datasets/aviation/out.csv")
        self.edit.textChanged.connect(self.parent.target_file_edit_changed)
        self.hbox_layout.addWidget(self.edit)

        self.button = QPushButton("Select File")
        self.button.clicked.connect(self.button_pressed)
        self.hbox_layout.addWidget(self.button)

        self.hbox_widget.setLayout(self.hbox_layout)
        self.layout.addWidget(self.hbox_widget)

        self.feedback_label = QLabel(" ")
        self.feedback_label.setFont(LABEL_FONT_ITALIC)
        self.feedback_label.setStyleSheet("color: red")
        self.layout.addWidget(self.feedback_label)

        self.setLayout(self.layout)

    def button_pressed(self, _):
        logger.info("Select where to save the resulting table.")
        path = str(QFileDialog.getSaveFileName(self, "Select where to save the resulting table.")[0])
        if path != "":
            self.edit.setText(path)
            self.parent.check_target_file()


class AttributesWidget(QWidget):
    """List of attributes."""

    def __init__(self, parent):
        super(AttributesWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 0)

        self.subheader = QLabel("3. Enter attributes to extract from the documents.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.label = QLabel("Each attribute must have a unique name. You may also provide example values for each "
                            "attribute.")
        self.label.setWordWrap(True)
        self.label.setFont(LABEL_FONT)
        self.layout.addWidget(self.label)

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout()
        self.list_layout.setContentsMargins(0, 10, 0, 10)

        self.attribute_widgets = [AttributeWidget(self)]
        for attribute_widget in self.attribute_widgets:
            self.list_layout.addWidget(attribute_widget)

        self.list_widget.setLayout(self.list_layout)
        self.layout.addWidget(self.list_widget)

        self.add_attribute_button = QPushButton("Add Attribute")
        self.add_attribute_button.clicked.connect(self.add_attribute_button_pressed)
        self.layout.addWidget(self.add_attribute_button)

        self.feedback_label = QLabel(" ")
        self.feedback_label.setFont(LABEL_FONT_ITALIC)
        self.layout.addWidget(self.feedback_label)

        self.setLayout(self.layout)
        self.layout.addStretch()

    def add_attribute_button_pressed(self):
        new_attribute_widget = AttributeWidget(self)
        self.attribute_widgets.append(new_attribute_widget)
        self.list_layout.addWidget(new_attribute_widget)
        self.list_layout.addStretch()
        self.layout.addStretch()
        self.parent.layout.addStretch()
        self.parent.attributes_edit_changed(None)
        # self.parent.setFixedSize(self.parent.sizeHint())

    def remove_attribute(self, attribute_widget):
        self.list_layout.removeWidget(attribute_widget)
        self.attribute_widgets.remove(attribute_widget)
        attribute_widget.deleteLater()
        self.list_layout.addStretch()
        self.layout.addStretch()
        self.parent.layout.addStretch()
        self.parent.attributes_edit_changed(None)
        # self.parent.setFixedSize(self.parent.sizeHint())


class AttributeWidget(QFrame):
    """One attribute."""

    def __init__(self, parent):
        super(AttributeWidget, self).__init__()
        self.parent = parent

        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)

        # add the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout()
        self.top_layout.setContentsMargins(0, 0, 0, 0)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("attribute name")
        self.name_edit.textChanged.connect(self.parent.parent.attributes_edit_changed)
        self.top_layout.addWidget(self.name_edit)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_button_pressed)
        self.top_layout.addWidget(self.remove_button)

        self.top_widget.setLayout(self.top_layout)
        self.layout.addWidget(self.top_widget)

        self.examples_widget = QWidget()
        self.examples_layout = QVBoxLayout()
        self.examples_layout.setContentsMargins(30, 0, 0, 0)

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout()
        self.list_layout.setContentsMargins(0, 0, 0, 0)

        self.example_widgets = [ExampleWidget(self)]
        for example_widget in self.example_widgets:
            self.list_layout.addWidget(example_widget)

        self.list_widget.setLayout(self.list_layout)
        self.examples_layout.addWidget(self.list_widget)

        self.add_example_button = QPushButton("Add Example Value")
        self.add_example_button.clicked.connect(self.add_example_button_pressed)
        self.examples_layout.addWidget(self.add_example_button)

        self.examples_widget.setLayout(self.examples_layout)
        self.layout.addWidget(self.examples_widget)

        self.setLayout(self.layout)

    def remove_example(self, example_widget):
        self.list_layout.removeWidget(example_widget)
        self.example_widgets.remove(example_widget)
        example_widget.deleteLater()
        self.list_layout.addStretch()
        self.layout.addStretch()
        self.parent.layout.addStretch()
        self.parent.parent.layout.addStretch()
        self.parent.parent.attributes_edit_changed(None)
        # self.parent.parent.setFixedSize(self.parent.parent.sizeHint())

    def remove_button_pressed(self, _):
        self.parent.remove_attribute(self)

    def add_example_button_pressed(self, _):
        new_example_widget = ExampleWidget(self)
        self.example_widgets.append(new_example_widget)
        self.list_layout.addWidget(new_example_widget)
        self.list_layout.addStretch()
        self.layout.addStretch()
        self.parent.layout.addStretch()
        self.parent.parent.layout.addStretch()
        self.parent.parent.attributes_edit_changed(None)
        # self.parent.parent.setFixedSize(self.parent.parent.sizeHint())


class ExampleWidget(QWidget):
    """One example value of an attribute."""

    def __init__(self, parent):
        super(ExampleWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("example value")
        self.value_edit.textChanged.connect(self.parent.parent.parent.attributes_edit_changed)
        self.layout.addWidget(self.value_edit)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_button_pressed)
        self.layout.addWidget(self.remove_button)

        self.setLayout(self.layout)

    def remove_button_pressed(self, _):
        self.parent.remove_example(self)


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
            QLabel("- Start matching.")
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


class OnlinePhaseWindow(QWidget):
    """Window of the online phase."""

    def __init__(self, parent):
        super(OnlinePhaseWindow, self).__init__()
        self.parent = parent

        self.online_phase_worker = None
        self.worker_thread = None

        self.setWindowTitle("ASET: Online Matching Phase")

        # set up the widgets
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

        self.attributes_widget = AttributesWidget(self)
        self.layout.addWidget(self.attributes_widget)

        self.matching_preparation_widget = MatchingPreparationWidget(self)
        self.layout.addWidget(self.matching_preparation_widget)

        # select directory
        self.setLayout(self.layout)
        self.layout.addStretch()

        # lock the window size
        self.setFixedSize(800, 1200)

        logger.debug("Initialized online phase window.")

    def source_file_edit_changed(self, _):
        self.source_file_widget.feedback_label.setStyleSheet("color: black")
        self.source_file_widget.feedback_label.setText(" ")

    def target_file_edit_changed(self, _):
        self.target_file_widget.feedback_label.setStyleSheet("color: black")
        self.target_file_widget.feedback_label.setText(" ")

    def attributes_edit_changed(self, _):
        self.attributes_widget.feedback_label.setStyleSheet("color: black")
        self.attributes_widget.feedback_label.setText(" ")

    def check_source_file(self):
        """Check if the preprocessed document collection file is valid."""
        file_path = self.source_file_widget.edit.text()
        logger.info(f"Check source file path '{file_path}'.")

        # check that the path leads to a file
        if not os.path.isfile(file_path):
            logger.error("The provided source file path is invalid!")
            self.source_file_widget.feedback_label.setStyleSheet("color: red")
            self.source_file_widget.feedback_label.setText("The provided source file path is invalid!")
            return False

        # try to json-parse the preprocessed document collection
        # try:
        #     with open(file_path, encoding="utf-8") as file:
        #         json_dict = json.load(file)
        #         num_docs = len(json_dict["documents"])
        #         num_extracts = sum(len(d["extractions"]) for d in json_dict["documents"])
        #         self.source_file_widget.feedback_label.setStyleSheet("color: black")
        #         self.source_file_widget.feedback_label.setText(
        #             f"Found a document collection with {num_docs} documents and {num_extracts} extractions."
        #         )
        # except JSONDecodeError:
        #     logger.error("The provided source file is not a valid JSON file!")
        #     self.source_file_widget.feedback_label.setStyleSheet("color: red")
        #     self.source_file_widget.feedback_label.setText("The provided source file is not a valid JSON file!")
        #     return False

        return True

    def check_target_file(self):
        """Check if the target file path is valid."""
        file_path = self.target_file_widget.edit.text()
        logger.info(f"Check target file path '{file_path}'.")

        # check that the path without the final part leads to a folder
        if not os.path.isdir("/".join(pathlib.Path(file_path).parts[:-1])):
            logger.error("The provided target file path is invalid!")
            self.target_file_widget.feedback_label.setStyleSheet("color: red")
            self.target_file_widget.feedback_label.setText("The provided target file path is invalid!")
            return False

        # check that the entire path is not a folder
        if os.path.isdir(file_path):
            logger.error("The provided target file path is invalid!")
            self.target_file_widget.feedback_label.setStyleSheet("color: red")
            self.target_file_widget.feedback_label.setText("The provided target file path is invalid!")
            return False

        if os.path.isfile(file_path):
            logger.info("Resulting table will overwrite existing file.")
            self.target_file_widget.feedback_label.setStyleSheet("color: black")
            self.target_file_widget.feedback_label.setText("Existing file will be overwritten.")

        return True

    def check_attributes(self):
        """Check that the attributes are valid."""
        attributes = set()

        for attribute_widget in self.attributes_widget.attribute_widgets:
            attribute_name = attribute_widget.name_edit.text()

            if attribute_name.strip() == "":
                logger.error("Attribute name cannot be an empty string!")
                self.attributes_widget.feedback_label.setStyleSheet("color: red")
                self.attributes_widget.feedback_label.setText("Attribute names cannot be empty strings!")
                return False

            if attribute_name in attributes:
                logger.error("Attribute name already exists!")
                self.attributes_widget.feedback_label.setStyleSheet("color: red")
                self.attributes_widget.feedback_label.setText("Duplicate attribute names are not allowed!")
                return False

            for example_widget in attribute_widget.example_widgets:
                example_value = example_widget.value_edit.text()

                if example_value.strip() == "":
                    logger.error("Empty example values are not allowed!")
                    self.attributes_widget.feedback_label.setStyleSheet("color: red")
                    self.attributes_widget.feedback_label.setText("Empty example values are not allowed!")
                    return False

            attributes.add(attribute_name)

        if len(attributes) == 0:
            logger.error("There must be at least one attribute!")
            self.attributes_widget.feedback_label.setStyleSheet("color: red")
            self.attributes_widget.feedback_label.setText("There must be at least one attribute!")
            return False

        self.attributes_widget.feedback_label.setStyleSheet("color: black")
        self.attributes_widget.feedback_label.setText(" ")
        return True

    def matching_preparation_finished(self):
        # update the GUI
        self.source_file_widget.setEnabled(True)
        self.target_file_widget.setEnabled(True)
        self.attributes_widget.setEnabled(True)
        self.matching_preparation_widget.start_button.setEnabled(True)

        self.matching_preparation_widget.progress_bar.setMinimum(0)
        self.matching_preparation_widget.progress_bar.setMaximum(100)
        self.matching_preparation_widget.progress_bar.setValue(100)

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
        for attribute_widget in self.attributes_widget.attribute_widgets:
            attribute_name = attribute_widget.name_edit.text()
            example_values = []
            for example_widget in attribute_widget.example_widgets:
                example_values.append(example_widget.value_edit.text())
            attributes[attribute_name] = example_values

        # start matching preparation
        self.online_phase_worker = OnlinePhaseWorker(
            source_path=self.source_file_widget.edit.text(),
            target_path=self.target_file_widget.edit.text(),
            attributes=attributes
        )

        self.worker_thread = QThread()
        self.online_phase_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.online_phase_worker.run)
        self.online_phase_worker.finished.connect(self.matching_preparation_finished)
        self.online_phase_worker.finished.connect(self.worker_thread.quit)
        self.online_phase_worker.finished.connect(self.online_phase_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.online_phase_worker.next.connect(self.matching_preparation_widget.next)
        self.online_phase_worker.progress.connect(self.matching_preparation_widget.progress)

        self.worker_thread.start()

    def closeEvent(self, event):
        """When window closed, go back to parent."""
        logger.info("Close online phase window.")
        self.parent.show()
