import json
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QLineEdit, QTextEdit

from aset_ui.style import SUBHEADER_FONT, HEADER_FONT, LABEL_FONT, CODE_FONT

logger = logging.getLogger(__name__)


class DocumentBaseViewer(QWidget):

    def __init__(self, main_window):
        super(DocumentBaseViewer, self).__init__()
        self.main_window = main_window

        self.setMinimumWidth(300)
        self.setMinimumHeight(100)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 10, 15, 10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)

        # header
        header = QLabel("Document Base")
        header.setFont(HEADER_FONT)
        self.layout.addWidget(header)

        # documents
        documents_widget = QWidget()
        documents_widget_layout = QVBoxLayout()
        documents_widget_layout.setContentsMargins(0, 10, 0, 10)
        documents_widget.setLayout(documents_widget_layout)
        self.layout.addWidget(documents_widget)

        documents_subheader = QLabel("Documents:")
        documents_subheader.setFont(SUBHEADER_FONT)
        documents_widget_layout.addWidget(documents_subheader)

        documents_information_widget = QWidget()
        documents_information_widget_layout = QGridLayout()
        documents_information_widget_layout.setContentsMargins(0, 0, 0, 0)
        documents_information_widget.setLayout(documents_information_widget_layout)
        documents_widget_layout.addWidget(documents_information_widget)

        num_documents_label = QLabel("number of documents:")
        num_documents_label.setFont(LABEL_FONT)
        documents_information_widget_layout.addWidget(num_documents_label, 0, 0)
        self.num_documents_value = QLineEdit()
        self.num_documents_value.setFont(CODE_FONT)
        self.num_documents_value.setReadOnly(True)
        documents_information_widget_layout.addWidget(self.num_documents_value, 0, 1)

        num_nuggets_label = QLabel("number of nuggets:")
        num_nuggets_label.setFont(LABEL_FONT)
        documents_information_widget_layout.addWidget(num_nuggets_label, 1, 0)
        self.num_nuggets_value = QLineEdit()
        self.num_nuggets_value.setFont(CODE_FONT)
        self.num_nuggets_value.setReadOnly(True)
        documents_information_widget_layout.addWidget(self.num_nuggets_value, 1, 1)

        # attributes
        attributes_widget = QWidget()
        attributes_widget_layout = QVBoxLayout()
        attributes_widget_layout.setContentsMargins(0, 10, 0, 10)
        attributes_widget.setLayout(attributes_widget_layout)
        self.layout.addWidget(attributes_widget)

        attributes_subheader = QLabel("Attributes:")
        attributes_subheader.setFont(SUBHEADER_FONT)
        attributes_widget_layout.addWidget(attributes_subheader)

        self.attribute_widgets = []
        attributes_list_widget = QWidget()
        self.attributes_list_widget_layout = QVBoxLayout()
        self.attributes_list_widget_layout.setContentsMargins(0, 0, 0, 0)
        attributes_list_widget.setLayout(self.attributes_list_widget_layout)
        attributes_widget_layout.addWidget(attributes_list_widget)

        logger.debug("Initialized DocumentBaseViewer.")

    def update_document_base(self, document_base):
        self.num_documents_value.setText(str(len(document_base.documents)))
        self.num_nuggets_value.setText(str(len(document_base.nuggets)))

        for widget in self.attribute_widgets:
            widget.hide()
            self.attributes_list_widget_layout.removeWidget(widget)
            widget.deleteLater()

        self.attribute_widgets = []
        for attribute in document_base.attributes:
            widget = QLineEdit()
            widget.setFont(CODE_FONT)
            widget.setReadOnly(True)
            widget.setText(attribute.name)
            self.attribute_widgets.append(widget)
            self.attributes_list_widget_layout.addWidget(widget)
        self.attributes_list_widget_layout.update()
        self.update()

    def enable_input(self):
        pass

    def disable_input(self):
        pass


class PreprocessingPhaseViewer(QWidget):

    def __init__(self, main_window):
        super(PreprocessingPhaseViewer, self).__init__()
        self.main_window = main_window

        self.setMinimumWidth(300)
        self.setMinimumHeight(100)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 10, 15, 10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)

        # header
        header = QLabel("Preprocessing Phase")
        header.setFont(HEADER_FONT)
        self.layout.addWidget(header)

        # configuration
        configuration_widget = QWidget()
        configuration_widget_layout = QVBoxLayout()
        configuration_widget_layout.setContentsMargins(0, 10, 0, 10)
        configuration_widget.setLayout(configuration_widget_layout)
        self.layout.addWidget(configuration_widget)

        configuration_subheader = QLabel("Configuration:")
        configuration_subheader.setFont(SUBHEADER_FONT)
        configuration_widget_layout.addWidget(configuration_subheader)

        self.configuration_edit = QTextEdit("")
        self.configuration_edit.setFont(CODE_FONT)
        self.configuration_edit.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
        self.configuration_edit.setLineWrapColumnOrWidth(10000)
        self.configuration_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.configuration_edit.setReadOnly(True)
        configuration_widget_layout.addWidget(self.configuration_edit)

        logger.debug("Initialized PreprocessingPhaseViewer.")

    def update_preprocessing_phase(self, preprocessing_phase):
        self.configuration_edit.setText(json.dumps(preprocessing_phase.to_config(), indent=2))

    def enable_input(self):
        pass

    def disable_input(self):
        pass


class MatchingPhaseViewer(QWidget):

    def __init__(self, main_window):
        super(MatchingPhaseViewer, self).__init__()
        self.main_window = main_window

        self.setMinimumWidth(300)
        self.setMinimumHeight(100)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 10, 15, 10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)

        # header
        header = QLabel("Matching Phase")
        header.setFont(HEADER_FONT)
        self.layout.addWidget(header)

        # configuration
        configuration_widget = QWidget()
        configuration_widget_layout = QVBoxLayout()
        configuration_widget_layout.setContentsMargins(0, 10, 0, 10)
        configuration_widget.setLayout(configuration_widget_layout)
        self.layout.addWidget(configuration_widget)

        configuration_subheader = QLabel("Configuration:")
        configuration_subheader.setFont(SUBHEADER_FONT)
        configuration_widget_layout.addWidget(configuration_subheader)

        self.configuration_edit = QTextEdit("")
        self.configuration_edit.setFont(CODE_FONT)
        self.configuration_edit.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
        self.configuration_edit.setLineWrapColumnOrWidth(10000)
        self.configuration_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.configuration_edit.setReadOnly(True)
        configuration_widget_layout.addWidget(self.configuration_edit)

        logger.debug("Initialized MatchingPhaseViewer.")

    def update_matching_phase(self, matching_phase):
        self.configuration_edit.setText(json.dumps(matching_phase.to_config(), indent=2))

    def enable_input(self):
        pass

    def disable_input(self):
        pass
