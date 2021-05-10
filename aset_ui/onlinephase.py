"""Online Phase: Match between document collection and query."""
import logging
import os

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog

from aset.extraction.extractionstage import ExtractionStage

logger = logging.getLogger(__name__)


class OnlinePhaseWindow(QWidget):
    """Window of the online phase."""

    def __init__(self, parent):
        super(OnlinePhaseWindow, self).__init__()
        self.parent = parent

        self.setWindowTitle("ASET: Online Matching Phase")

        # set up the widgets
        self.layout = QVBoxLayout()

        # header
        self.header = QLabel("ASET: Online Matching Phase")
        header_font = self.header.font()
        header_font.setPointSize(20)
        header_font.setWeight(100)
        self.header.setFont(header_font)
        self.layout.addWidget(self.header)

        # widgets
        self.open_document_collection_widget = OpenDocumentCollectionWidget(self)
        self.layout.addWidget(self.open_document_collection_widget)

        self.enter_attributes_widget = EnterAttributesWidget(self)

        # select directory
        self.setLayout(self.layout)
        self.layout.addStretch()

        # lock the window size
        self.setFixedSize(QSize(800, 400))

        logger.debug("Initialized online phase window.")

    def closeEvent(self, event):
        """When window closed, go back to parent."""
        logger.info("Close online phase window.")
        self.parent.show()


class OpenDocumentCollectionWidget(QWidget):

    def __init__(self, parent):
        super(OpenDocumentCollectionWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()

        # subheader
        self.subheader = QLabel("1. Open a preprocessed document collection.")
        subheader_font = self.subheader.font()
        subheader_font.setPointSize(12)
        subheader_font.setWeight(60)
        self.subheader.setFont(subheader_font)
        self.layout.addWidget(self.subheader)

        # select file hbox
        self.select_file_hbox_widget = QWidget()
        self.select_file_hbox_layout = QHBoxLayout()
        self.select_file_hbox_layout.setContentsMargins(0, 0, 0, 0)

        self.select_file_edit = QLineEdit()
        self.select_file_edit.setPlaceholderText("Select File")
        self.select_file_hbox_layout.addWidget(self.select_file_edit)

        self.select_file_button = QPushButton("Select File")
        self.select_file_button.clicked.connect(self.select_file)
        self.select_file_hbox_layout.addWidget(self.select_file_button)

        self.select_file_hbox_widget.setLayout(self.select_file_hbox_layout)
        self.layout.addWidget(self.select_file_hbox_widget)

        self.open_document_collection_button = QPushButton("Open Preprocessed Document Collection")
        self.open_document_collection_button.clicked.connect(self.open_preprocessed_document_collection)
        self.layout.addWidget(self.open_document_collection_button)

        # feedback label
        self.select_file_feedback_label = QLabel(" ")
        self.layout.addWidget(self.select_file_feedback_label)

        self.setLayout(self.layout)
        self.layout.addStretch()

    def open_preprocessed_document_collection(self, _):
        if not os.path.isfile(self.select_file_edit.text()):
            self.select_file_feedback_label.setText("Path does not lead to a file!")
            return

        with open(self.select_file_edit.text(), encoding="utf-8") as file:
            self.parent.extraction_stage = ExtractionStage.from_json_str(file.read())

        self.hide()
        self.parent.enter_attributes_widget.show()

    def select_file(self, _):
        self.select_file_edit.setText(str(QFileDialog.getOpenFileUrl(self, "Select Document Collection File")))


class EnterAttributesWidget(QWidget):

    def __init__(self, parent):
        super(EnterAttributesWidget, self).__init__()
        self.parent = parent

        # set up the widgets
        self.layout = QVBoxLayout()

        # subheader
        self.subheader = QLabel("2. Enter attributes.")
        subheader_font = self.subheader.font()
        subheader_font.setPointSize(12)
        subheader_font.setWeight(60)
        self.subheader.setFont(subheader_font)
        self.layout.addWidget(self.subheader)

        # attribute widgets
        self.attribute_widgets = []
        for attribute_widget in self.attribute_widgets:
            self.layout.addWidget(attribute_widget)

        self.add_attribute_button = QPushButton("Add Attribute")
        self.add_attribute_button.clicked.connect(self.add_attribute_button_clicked)
        self.layout.addWidget(self.add_attribute_button)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.layout.addWidget(self.continue_button)

        # feedback label
        self.feedback_label = QLabel(" ")
        self.layout.addWidget(self.feedback_label)

        self.setLayout(self.layout)
        self.layout.addStretch()

    def continue_button_clicked(self, _):
        pass

    def add_attribute_button_clicked(self):
        new_attribute_widget = AttributeWidget(self)
        self.attribute_widgets.append(new_attribute_widget)
        self.layout.addWidget(new_attribute_widget)
        self.layout.addStretch()

    def remove_attribute(self, attribute_widget):
        self.layout.removeWidget(attribute_widget)
        self.attribute_widgets.remove(attribute_widget)
        self.layout.addStretch()


class AttributeWidget(QWidget):

    def __init__(self, parent):
        super(AttributeWidget, self).__init__()
        self.parent = parent

        # add the widgets
        self.layout = QHBoxLayout()

        self.attribute_name_edit = QLineEdit()
        self.attribute_name_edit.setPlaceholderText("attribute name")
        self.layout.addWidget(self.attribute_name_edit)

        self.delete_attribute_button = QPushButton("x")
        self.delete_attribute_button.clicked.connect(self.delete_attribute)
        self.layout.addWidget(self.delete_attribute_button)

        self.setLayout(self.layout)
        self.layout.addStretch()

    def delete_attribute(self, _):
        self.parent.remove_attribute(self)
