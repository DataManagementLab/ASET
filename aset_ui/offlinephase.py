"""Offline Phase: Preprocess a document collection."""
import glob
import logging
import os
import pathlib

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, QLineEdit, QProgressBar

from aset.embedding.aggregation import ExtractionEmbeddingMethod
from aset.extraction.common import Document
from aset.extraction.extractionstage import ExtractionStage
from aset.extraction.extractors import StanzaExtractor
from aset.extraction.processors import StanfordCoreNLPDateTimeProcessor, StanfordCoreNLPNumberProcessor, \
    StanfordCoreNLPStringProcessor
from aset_ui.util import SUBHEADER_FONT, HEADER_FONT, LABEL_FONT, LABEL_FONT_BOLD, LABEL_FONT_ITALIC

logger = logging.getLogger(__name__)


class OfflinePhaseWorker(QObject):
    """Worker that executes the offline phase."""

    finished: pyqtSignal = pyqtSignal()
    progress: pyqtSignal = pyqtSignal(float)
    next: pyqtSignal = pyqtSignal()

    def __init__(self, source_path: str, target_path: str):
        """
        Initialize the offline phase worker with paths to the folder containing the documents and the file where to save
        the preprocessed document collection.

        :param source_path: path to the folder containing the documents
        :param target_path: path to the file where the preprocessed document collection should be stored
        """
        super(OfflinePhaseWorker, self).__init__()
        self.source_path: str = source_path
        self.target_path: str = target_path
        self.extraction_stage: ExtractionStage or None = None

    def run(self):
        """Main code of the offline phase."""

        # load the document collection
        logger.info(f"Load document collection from '{self.source_path}'.")
        self.next.emit()

        self.progress.emit(-1.)

        file_paths = glob.glob(self.source_path + "/*.txt")
        documents = []
        for ix, file_path in enumerate(file_paths):
            if ix % 10 == 0:
                self.progress.emit(ix / len(file_paths))
            with open(file_path, encoding="utf-8") as file:
                documents.append(Document(file.read()))

        self.next.emit()

        # load the extraction stage
        logger.info("Load the extraction stage.")
        self.progress.emit(-1.)

        extractors = [StanzaExtractor()]
        for ix, extractor in enumerate(extractors):
            extractor.status_callback = lambda x, a=ix, b=len(extractors): self.progress.emit((x + a) / b)

        processors = [
            StanfordCoreNLPDateTimeProcessor(),
            StanfordCoreNLPNumberProcessor(),
            StanfordCoreNLPStringProcessor()
        ]
        for ix, processor in enumerate(processors):
            processor.status_callback = lambda x, a=ix, b=len(processors): self.progress.emit((x + a) / b)

        embedding_method = ExtractionEmbeddingMethod()
        embedding_method.status_callback = lambda x: self.progress.emit(x)

        self.extraction_stage = ExtractionStage(
            documents=documents,
            extractors=extractors,
            processors=processors,
            embedding_method=embedding_method
        )
        self.next.emit()

        # derive extractions
        logger.info("Derive extractions.")
        self.progress.emit(0.)
        self.extraction_stage.derive_extractions()
        self.progress.emit(1.)
        self.next.emit()

        # determine values
        logger.info("Determine values.")
        self.progress.emit(0.)
        self.extraction_stage.determine_values()
        self.progress.emit(1.)
        self.next.emit()

        # compute extraction embeddings
        logger.info("Compute extraction embeddings.")
        self.progress.emit(0.)
        self.extraction_stage.compute_extraction_embeddings()
        self.progress.emit(1.)
        self.next.emit()

        # store the preprocessed document collection
        logger.info(f"Store preprocessed document collection to {self.target_path}")
        self.progress.emit(-1.)
        with open(self.target_path, "w", encoding="utf-8") as file:
            file.write(self.extraction_stage.json_str)
        print("...")
        self.next.emit()

        logger.info("All done.")
        self.finished.emit()


class SourceDirectoryWidget(QWidget):
    """Widget to select the source directory."""

    def __init__(self, parent):
        super(SourceDirectoryWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 0)

        self.subheader = QLabel("1. Choose from where to load the document collection.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.label = QLabel("ASET expects the document collection to be a directory that contains one .txt file of "
                            "raw text for each document.")
        self.label.setFont(LABEL_FONT)
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select Directory")
        self.edit.textChanged.connect(self.parent.source_directory_edit_changed)
        self.hbox_layout.addWidget(self.edit)

        self.button = QPushButton("Select Directory")
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
        """Select directory button pressed."""
        logger.info("Select document collection directory")
        path = str(QFileDialog.getExistingDirectory(self, "Choose from where to load the document collection"))
        if path != "":
            self.edit.setText(path)
            self.parent.check_source_directory()


class TargetFileWidget(QWidget):
    """Widget to select the target file."""

    def __init__(self, parent):
        super(TargetFileWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 0)

        self.subheader = QLabel("2. Choose where to store the preprocessed document collection.")
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
        """Select file button pressed."""
        logger.info("Select preprocessed document collection file.")
        path = str(QFileDialog.getSaveFileName(self, "Choose where to store the preprocessed document collection")[0])
        if path != "":
            self.edit.setText(path)
            self.parent.check_target_file()


class PreprocessingWidget(QWidget):
    """Widget to start and monitor the preprocessing."""

    def __init__(self, parent):
        super(PreprocessingWidget, self).__init__()
        self.parent = parent
        self.current_step = -1

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 10)

        self.subheader = QLabel("3. Preprocess the document collection.")
        self.subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.subheader)

        self.steps_widget = QWidget()
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setContentsMargins(30, 5, 0, 10)

        self.steps = [
            QLabel("- Load document collection."),
            QLabel("- Load extraction stage."),
            QLabel("- Derive extractions."),
            QLabel("- Determine extraction values."),
            QLabel("- Compute extraction embeddings."),
            QLabel("- Store preprocessed extraction stage.")
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
        self.start_button.setText("Start Preprocessing")
        self.start_button.clicked.connect(self.parent.start_preprocessing)
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


class OfflinePhaseWindow(QWidget):
    """Window of the offline phase."""

    def __init__(self, parent):
        super(OfflinePhaseWindow, self).__init__()
        self.parent = parent

        self.offline_phase_worker = None
        self.worker_thread = None

        self.setWindowTitle("ASET: Offline Preprocessing Phase")

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 15, 20, 15)

        # header
        self.header = QLabel("ASET: Offline Preprocessing Phase")
        self.header.setFont(HEADER_FONT)
        self.layout.addWidget(self.header)

        # select document collection directory
        self.source_directory_widget = SourceDirectoryWidget(self)
        self.layout.addWidget(self.source_directory_widget)

        # select file for preprocessed document collection
        self.target_file_widget = TargetFileWidget(self)
        self.layout.addWidget(self.target_file_widget)

        # preprocessing button and monitoring
        self.preprocessing_widget = PreprocessingWidget(self)
        self.layout.addWidget(self.preprocessing_widget)

        self.setLayout(self.layout)
        self.layout.addStretch()

        # lock the window size
        self.setFixedSize(self.sizeHint())

        logger.debug("Initialized offline phase window.")

    def source_directory_edit_changed(self, _):
        self.source_directory_widget.feedback_label.setStyleSheet("color: black")
        self.source_directory_widget.feedback_label.setText(" ")

    def target_file_edit_changed(self, _):
        self.target_file_widget.feedback_label.setStyleSheet("color: black")
        self.target_file_widget.feedback_label.setText(" ")

    def closeEvent(self, _):
        """When window closed, go back to parent."""
        logger.info("Close offline phase window.")
        self.parent.show()

    def check_source_directory(self):
        """Check if the source directory path is valid."""
        directory_path = self.source_directory_widget.edit.text()
        logger.info(f"Check source path '{directory_path}'.")

        # check that the path leads to a directory
        if not os.path.isdir(directory_path):
            logger.error("The provided source path is invalid!")
            self.source_directory_widget.feedback_label.setStyleSheet("color: red")
            self.source_directory_widget.feedback_label.setText("The provided source path is invalid!")
            return False

        file_paths = glob.glob(directory_path + "/*.txt")
        num_documents = len(file_paths)

        # check that there are valid documents
        if num_documents == 0:
            logger.error("There are no valid documents in the directory!")
            self.source_directory_widget.feedback_label.setStyleSheet("color: red")
            self.source_directory_widget.feedback_label.setText(f"There are no valid documents in the directory!")
            return False

        logger.info(f"Found {num_documents} documents in the directory.")
        self.source_directory_widget.feedback_label.setStyleSheet("color: black")
        self.source_directory_widget.feedback_label.setText(f"Found {num_documents} documents.")
        return True

    def check_target_file(self):
        """Check if the target file path is valid."""
        file_path = self.target_file_widget.edit.text()
        logger.info(f"Check target path '{file_path}'.")

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
            logger.info("Preprocessed document collection will overwrite existing file.")
            self.target_file_widget.feedback_label.setStyleSheet("color: black")
            self.target_file_widget.feedback_label.setText("Existing file will be overwritten.")

        return True

    def preprocessing_finished(self):
        # update the UI
        self.source_directory_widget.setEnabled(True)
        self.target_file_widget.setEnabled(True)
        self.preprocessing_widget.start_button.setEnabled(True)

        self.preprocessing_widget.progress_bar.setMinimum(0)
        self.preprocessing_widget.progress_bar.setMaximum(100)
        self.preprocessing_widget.progress_bar.setValue(100)

    def start_preprocessing(self, _):
        """Start preprocessing if the directory and file are valid."""
        logger.info("Start preprocessing.")

        # check that the source directory is valid
        if not self.check_source_directory():
            return

        # check that the target file is valid
        if not self.check_target_file():
            return

        # update the UI
        self.source_directory_widget.setEnabled(False)
        self.target_file_widget.setEnabled(False)
        self.preprocessing_widget.start_button.setEnabled(False)

        # start preprocessing
        self.offline_phase_worker = OfflinePhaseWorker(
            source_path=self.source_directory_widget.edit.text(),
            target_path=self.target_file_widget.edit.text()
        )

        self.worker_thread = QThread()
        self.offline_phase_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.offline_phase_worker.run)
        self.offline_phase_worker.finished.connect(self.preprocessing_finished)
        self.offline_phase_worker.finished.connect(self.worker_thread.quit)
        self.offline_phase_worker.finished.connect(self.offline_phase_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.offline_phase_worker.next.connect(self.preprocessing_widget.next)
        self.offline_phase_worker.progress.connect(self.preprocessing_widget.progress)

        self.worker_thread.start()
