import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog

from aset_ui.util import SUBHEADER_FONT, LABEL_FONT, LABEL_FONT_ITALIC

logger = logging.getLogger(__name__)


class SourceFileWidget(QWidget):
    """Widget to select the source file."""

    def __init__(self, parent):
        super(SourceFileWidget, self).__init__(parent)
        self._parent = parent

        # layout, header, and label
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 10, 0, 10)

        self._subheader = QLabel("1. Open a preprocessed document collection.")
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._label = QLabel("ASET stores the preprocessed document collection as a .json file.")
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
        self._edit.setPlaceholderText("source file path")
        self._edit.textChanged.connect(self.filepath_changed)
        self._file_path_layout.addWidget(self._edit)

        self._button = QPushButton("Select File")
        self._button.clicked.connect(self.button_pressed)
        self._file_path_layout.addWidget(self._button)

        # feedback label
        self._feedback_label = QLabel(" ")
        self._feedback_label.setFont(LABEL_FONT_ITALIC)
        self._layout.addWidget(self._feedback_label)

    def button_pressed(self):
        path = str(QFileDialog.getOpenFileName(self, "Select preprocessed document collection file")[0])
        if path != "":
            self._edit.setText(path)
            self._parent.check_source_file()

    def filepath_changed(self):
        self._feedback_label.setStyleSheet("color: black")
        self._feedback_label.setText(" ")

    def give_feedback(self, feedback):
        self._feedback_label.setStyleSheet("color: red")
        self._feedback_label.setText(feedback)

    def get_filepath(self):
        return self._edit.text()


class TargetFileWidget(QWidget):
    """Widget to select the target file."""

    def __init__(self, parent):
        super(TargetFileWidget, self).__init__(parent)
        self._parent = parent

        # layout, header, and label
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(self._layout)

        self._subheader = QLabel("2. Choose where to store the resulting table.")
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._label = QLabel("ASET stores the resulting table as a .csv file.")
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
        self._edit.setPlaceholderText("target file path")
        self._edit.textChanged.connect(self.filepath_changed)
        self._file_path_layout.addWidget(self._edit)

        self._button = QPushButton("Select File")
        self._button.clicked.connect(self.button_pressed)
        self._file_path_layout.addWidget(self._button)

        # feedback label
        self._feedback_label = QLabel(" ")
        self._feedback_label.setFont(LABEL_FONT_ITALIC)
        self._layout.addWidget(self._feedback_label)

    def button_pressed(self):
        path = str(QFileDialog.getSaveFileName(self, "Select where to save the resulting table.")[0])
        if path != "":
            self._edit.setText(path)
            self._parent.check_target_file()

    def filepath_changed(self):
        self._feedback_label.setStyleSheet("color: black")
        self._feedback_label.setText(" ")

    def give_feedback(self, feedback):
        self._feedback_label.setStyleSheet("color: red")
        self._feedback_label.setText(feedback)

    def get_filepath(self):
        return self._edit.text()
