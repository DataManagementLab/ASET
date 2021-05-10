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

logger = logging.getLogger(__name__)


class OfflinePhaseWorker(QObject):
    """Worker that executes the offline phase."""

    finished = pyqtSignal()
    progress = pyqtSignal(str, float)

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

        self.progress.emit("Loading document collection...", 0.)

        file_paths = glob.glob(self.source_path + "/*.txt")
        documents = []
        for ix, file_path in enumerate(file_paths):
            if ix % 10 == 0:
                self.progress.emit("Loading document collection...", ix / len(file_paths))
            with open(file_path, encoding="utf-8") as file:
                documents.append(Document(file.read()))

        self.progress.emit("Loading document collection...", 1.)

        # load the extraction stage
        logger.info("Load the extraction stage.")
        self.progress.emit("Loading extraction stage (extractors)...", 0.)

        extractors = [StanzaExtractor()]
        for ix, extractor in enumerate(extractors):
            extractor.status_callback = lambda x: self.progress.emit(
                f"Deriving extractions ({ix}/{len(extractors)})...", x
            )
        self.progress.emit("Loading extraction stage (processors)...", 0.2)

        processors = [
            StanfordCoreNLPDateTimeProcessor(),
            StanfordCoreNLPNumberProcessor(),
            StanfordCoreNLPStringProcessor()
        ]
        for ix, processor in enumerate(processors):
            processor.status_callback = lambda x: self.progress.emit(
                f"Determining values ({ix}/{len(processors)})...", x
            )
        self.progress.emit("Loading extraction stage (extraction embedding method .. 'may take a while')...", 0.4)

        embedding_method = ExtractionEmbeddingMethod()
        embedding_method.status_callback = lambda x: self.progress.emit(
            "Computing extraction embeddings...", x
        )

        self.extraction_stage = ExtractionStage(
            documents=documents,
            extractors=extractors,
            processors=processors,
            embedding_method=embedding_method
        )
        self.progress.emit("Loading extraction stage...", 1.)

        # derive extractions
        logger.info("Derive extractions.")
        self.progress.emit(f"Deriving extractions (0/{len(extractors)})...", 0.)
        self.extraction_stage.derive_extractions()
        self.progress.emit(f"Deriving extractions ({len(extractors)}/{len(extractors)})...", 1.)

        # determine values
        logger.info("Determine values.")
        self.progress.emit(f"Determining values (0/{len(processors)})...", 0.)
        self.extraction_stage.determine_values()
        self.progress.emit(f"Determining values ({len(processors)}/{len(processors)})...", 1.)

        # compute extraction embeddings
        logger.info("Compute extraction embeddings.")
        self.progress.emit("Computing extraction embeddings...", 0.)
        self.extraction_stage.compute_extraction_embeddings()
        self.progress.emit("Computing extraction embeddings...", 1.)

        # store the preprocessed document collection
        logger.info(f"Store preprocessed document collection to {self.target_path}")
        self.progress.emit("Storing preprocessed document collection...", 0.)
        with open(self.target_path, "w", encoding="utf-8") as file:
            file.write(self.extraction_stage.json_str)
        self.progress.emit("Storing preprocessed document collection...", 1.)

        self.finished.emit()


class SourceDirectoryWidget(QWidget):
    """Widget to select the source directory."""

    def __init__(self, parent):
        super(SourceDirectoryWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 10)

        self.header = QLabel("Choose from where to load the document collection:")
        header_font = self.header.font()
        header_font.setPointSize(11)
        header_font.setWeight(60)
        self.header.setFont(header_font)
        self.layout.addWidget(self.header)

        self.label = QLabel(
            "ASET expects the document collection to be a directory that contains one .txt file of raw text for each document."
        )
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select Directory")
        self.edit.setText(r"C:\Users\micha\Code\ASET\datasets\aviation\raw-documents")
        self.edit.textChanged.connect(self.parent.source_directory_edit_changed)
        self.hbox_layout.addWidget(self.edit)

        self.button = QPushButton("Select Directory")
        self.button.clicked.connect(self.button_pressed)
        self.hbox_layout.addWidget(self.button)

        self.hbox_widget.setLayout(self.hbox_layout)
        self.layout.addWidget(self.hbox_widget)

        self.setLayout(self.layout)

    def button_pressed(self, _):
        """Select directory button pressed."""
        logger.info("Select document collection directory")
        directory_path = str(
            QFileDialog.getExistingDirectory(self, "Choose from where to load the document collection"))
        self.edit.setText(directory_path)


class TargetFileWidget(QWidget):
    """Widget to select the target file."""

    def __init__(self, parent):
        super(TargetFileWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 10, 0, 10)

        self.header = QLabel("Choose where to store the preprocessed document collection:")
        header_font = self.header.font()
        header_font.setPointSize(11)
        header_font.setWeight(60)
        self.header.setFont(header_font)
        self.layout.addWidget(self.header)

        self.label = QLabel(
            "ASET stores the preprocessed document collection as a json file."
        )
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select File")
        self.edit.setText(r"C:/Users/micha/Code/ASET/datasets/aviation/out.json")
        self.edit.textChanged.connect(self.parent.target_file_edit_changed)
        self.hbox_layout.addWidget(self.edit)

        self.button = QPushButton("Select File")
        self.button.clicked.connect(self.button_pressed)
        self.hbox_layout.addWidget(self.button)

        self.hbox_widget.setLayout(self.hbox_layout)
        self.layout.addWidget(self.hbox_widget)

        self.setLayout(self.layout)

    def button_pressed(self, _):
        """Select file button pressed."""
        logger.info("Select preprocessed document collection file.")
        file_path = str(
            QFileDialog.getSaveFileName(self, "Choose where to store the preprocessed document collection")[0])
        self.edit.setText(file_path)


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

        # header
        self.header = QLabel("ASET: Offline Preprocessing Phase")
        header_font = self.header.font()
        header_font.setPointSize(20)
        header_font.setWeight(100)
        self.header.setFont(header_font)
        self.layout.addWidget(self.header)

        # select document collection directory
        self.source_directory_widget = SourceDirectoryWidget(self)
        self.layout.addWidget(self.source_directory_widget)

        # select file for preprocessed document collection
        self.target_file_widget = TargetFileWidget(self)
        self.layout.addWidget(self.target_file_widget)

        # start preprocessing button
        self.start_preprocessing_button = QPushButton()
        self.start_preprocessing_button.setText("Start Preprocessing")
        self.start_preprocessing_button.clicked.connect(self.start_preprocessing)
        self.layout.addWidget(self.start_preprocessing_button)

        # preprocessing progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)

        # feedback label
        self.feedback_label = QLabel(" ")
        self.layout.addWidget(self.feedback_label)

        self.setLayout(self.layout)
        self.layout.addStretch()

        # lock the window size
        self.setFixedSize(self.sizeHint())

        logger.debug("Initialized offline phase window.")

    def source_directory_edit_changed(self, _):
        self.feedback_label.setText(" ")

    def target_file_edit_changed(self, _):
        self.feedback_label.setText(" ")

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
            self.feedback_label.setText("The provided source path is invalid!")
            return False

        file_paths = glob.glob(directory_path + "/*.txt")
        num_documents = len(file_paths)

        # check that there are valid documents
        if num_documents == 0:
            logger.error("There are no valid documents in the directory!")
            self.feedback_label.setText(f"There are no valid documents in the directory!")
            return False

        logger.info(f"Found {num_documents} documents in the directory.")
        self.feedback_label.setText(f"Found {num_documents} documents.")
        return True

    def check_target_file(self):
        """Check if the target file path is valid."""
        file_path = self.target_file_widget.edit.text()
        logger.info(f"Check target path '{file_path}'.")

        # check that the path without the final part leads to a folder
        if not os.path.isdir("/".join(pathlib.Path(file_path).parts[:-1])):
            logger.error("The provided target file path is invalid!")
            self.feedback_label.setText("The provided target file path is invalid!")
            return False

        return True

    def preprocessing_progress(self, description: str, fraction_done: float):
        self.feedback_label.setText(description)
        self.progress_bar.setValue(int(fraction_done * 100))

    def preprocessing_finished(self):
        # update the UI
        self.source_directory_widget.edit.setEnabled(True)
        self.source_directory_widget.button.setEnabled(True)
        self.target_file_widget.edit.setEnabled(True)
        self.target_file_widget.button.setEnabled(True)
        self.start_preprocessing_button.setEnabled(True)

        self.progress_bar.setValue(100)
        self.feedback_label.setText("Preprocessing document collection finished!")

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
        self.source_directory_widget.edit.setEnabled(False)
        self.source_directory_widget.button.setEnabled(False)
        self.target_file_widget.edit.setEnabled(False)
        self.target_file_widget.button.setEnabled(False)
        self.start_preprocessing_button.setEnabled(False)

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
        self.offline_phase_worker.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.offline_phase_worker.progress.connect(self.preprocessing_progress)

        print("Start thread.")
        self.worker_thread.start()
        print("Thread started.")
