from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, \
    QVBoxLayout, QWidget

from aset_ui.style import BUTTON_FONT, CODE_FONT, CODE_FONT_BOLD, HEADER_FONT, LABEL_FONT, SUBHEADER_FONT


class DocumentBaseViewerWidget(QWidget):
    def __init__(self, main_window):
        super(DocumentBaseViewerWidget, self).__init__()
        self.main_window = main_window

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(30)

        # header
        self.header = QLabel("Document Base")
        self.header.setFont(HEADER_FONT)
        self.layout.addWidget(self.header)

        # documents
        self.documents = QWidget()
        self.documents_layout = QVBoxLayout(self.documents)
        self.documents_layout.setContentsMargins(0, 0, 0, 0)
        self.documents_layout.setSpacing(10)
        self.layout.addWidget(self.documents)

        self.documents_subheader = QLabel("Documents:")
        self.documents_subheader.setFont(SUBHEADER_FONT)
        self.documents_layout.addWidget(self.documents_subheader)

        self.num_documents = QLabel("number of documents: -")
        self.num_documents.setFont(LABEL_FONT)
        self.documents_layout.addWidget(self.num_documents)

        self.num_nuggets = QLabel("number of nuggets: -")
        self.num_nuggets.setFont(LABEL_FONT)
        self.documents_layout.addWidget(self.num_nuggets)

        # attributes
        self.attributes = QWidget()
        self.attributes_layout = QVBoxLayout(self.attributes)
        self.attributes_layout.setContentsMargins(0, 0, 0, 0)
        self.attributes_layout.setSpacing(10)
        self.layout.addWidget(self.attributes)

        self.attributes_subheader = QLabel("Attributes:")
        self.attributes_subheader.setFont(SUBHEADER_FONT)
        self.attributes_layout.addWidget(self.attributes_subheader)

        self.attribute_widgets = []
        self.num_visible_attribute_widgets = 0
        self.attribute_list = QWidget()
        self.attribute_list_layout = QVBoxLayout(self.attribute_list)
        self.attribute_list_layout.setContentsMargins(0, 0, 10, 0)
        self.attribute_list_layout.setSpacing(10)
        self.attribute_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.attribute_scroll_area = QScrollArea()
        self.attribute_scroll_area.setWidgetResizable(True)
        self.attribute_scroll_area.setFrameStyle(0)
        self.attribute_scroll_area.setWidget(self.attribute_list)
        self.attributes_layout.addWidget(self.attribute_scroll_area)

    def update_document_base(self, document_base):
        # update documents
        self.num_documents.setText(f"number of documents: {len(document_base.documents)}")
        self.num_nuggets.setText(f"number of nuggets: {len(document_base.nuggets)}")

        # update attributes
        # make sure that there are enough attribute widgets
        while len(document_base.attributes) > len(self.attribute_widgets):
            self.attribute_widgets.append(AttributeWidget(self))

        # make sure that the correct number of attribute widgets is shown
        while len(document_base.attributes) > self.num_visible_attribute_widgets:
            widget = self.attribute_widgets[self.num_visible_attribute_widgets]
            self.attribute_list_layout.addWidget(widget)
            self.num_visible_attribute_widgets += 1
        while len(document_base.attributes) < self.num_visible_attribute_widgets:
            widget = self.attribute_widgets[self.num_visible_attribute_widgets - 1]
            widget.hide()
            self.attribute_list_layout.removeWidget(widget)
            self.num_visible_attribute_widgets -= 1

        # update the attribute widgets
        max_attribute_name_len = max([len(attribute.name) for attribute in document_base.attributes])
        for attribute, widget in zip(document_base.attributes, self.attribute_widgets[: len(document_base.attributes)]):
            widget.update_attribute(attribute, document_base, max_attribute_name_len)


class AttributeWidget(QFrame):
    def __init__(self, document_base_viewer):
        super(AttributeWidget, self).__init__(document_base_viewer)
        self.document_base_viewer = document_base_viewer
        self.attribute = None

        self.setFixedHeight(30)
        self.setStyleSheet("background-color: white")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 0, 0)
        self.layout.setSpacing(30)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.attribute_name = QLabel()
        self.attribute_name.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.attribute_name)

        self.num_matched = QLabel("matches: -")
        self.num_matched.setFont(CODE_FONT)
        self.layout.addWidget(self.num_matched)

    def update_attribute(self, attribute, document_base, max_attribute_name_len):
        self.attribute = attribute

        self.attribute_name.setText(attribute.name + (" " * (max_attribute_name_len - len(attribute.name))))
        num_matches = 0
        for document in document_base.documents:
            if attribute.name in document.attribute_mappings.keys() and \
                    document.attribute_mappings[attribute.name] != []:
                num_matches += 1
            self.num_matched.setText(f"matches: {num_matches}")


class CreateDocumentBaseWidget(QWidget):
    def __init__(self, main_window) -> None:
        super(CreateDocumentBaseWidget, self).__init__()
        self.main_window = main_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        # header
        self.header = QLabel("Create Document Base")
        self.header.setFont(HEADER_FONT)
        self.layout.addWidget(self.header)

        self.documents_subheader = QLabel("Documents:")
        self.documents_subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.documents_subheader)

        self.documents_explanation = QLabel(
            "Enter the path of the directory that contains the documents as .txt files."
        )
        self.documents_explanation.setFont(LABEL_FONT)
        self.layout.addWidget(self.documents_explanation)

        self.path_wrapper_widget = QWidget()
        self.path_wrapper_layout = QHBoxLayout(self.path_wrapper_widget)
        self.path_wrapper_layout.setContentsMargins(0, 0, 10, 0)
        self.layout.addWidget(self.path_wrapper_widget)

        self.path_widget = QFrame()
        self.path_layout = QHBoxLayout(self.path_widget)
        self.path_layout.setContentsMargins(10, 0, 10, 0)
        self.path_layout.setSpacing(10)
        self.path_widget.setStyleSheet("background-color: white")
        self.path_widget.setFixedHeight(40)
        self.path_wrapper_layout.addWidget(self.path_widget)

        self.path = QLineEdit()
        self.path.setFont(CODE_FONT_BOLD)
        self.path.setStyleSheet("border: none")
        self.path_layout.addWidget(self.path)

        self.edit_path_button = QPushButton()
        self.edit_path_button.setIcon(QIcon("aset_ui/resources/folder.svg"))
        self.edit_path_button.setFlat(True)
        self.edit_path_button.clicked.connect(self._edit_path_button_clicked)
        self.path_layout.addWidget(self.edit_path_button)

        self.attributes_subheader = QLabel("Attributes:")
        self.attributes_subheader.setFont(SUBHEADER_FONT)
        self.layout.addWidget(self.attributes_subheader)

        self.labels_explanation = QLabel("Enter the attribute names.")
        self.labels_explanation.setFont(LABEL_FONT)
        self.layout.addWidget(self.labels_explanation)

        self.create_attribute_widgets = []
        self.num_visible_create_attribute_widgets = 0
        self.create_attribute_list = QWidget()
        self.create_attribute_list_layout = QVBoxLayout(self.create_attribute_list)
        self.create_attribute_list_layout.setContentsMargins(0, 0, 10, 0)
        self.create_attribute_list_layout.setSpacing(10)
        self.create_attribute_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.create_attribute_scroll_area = QScrollArea()
        self.create_attribute_scroll_area.setWidgetResizable(True)
        self.create_attribute_scroll_area.setFrameStyle(0)
        self.create_attribute_scroll_area.setWidget(self.create_attribute_list)
        self.layout.addWidget(self.create_attribute_scroll_area)

        self.create_attribute_button = QPushButton("New Attribute")
        self.create_attribute_button.setFont(BUTTON_FONT)
        self.create_attribute_button.clicked.connect(self._create_attribute_button_clicked)
        self.create_attribute_list_layout.addWidget(self.create_attribute_button)

        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 10, 0)
        self.buttons_layout.setSpacing(10)
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout.addWidget(self.buttons_widget)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(BUTTON_FONT)
        self.cancel_button.clicked.connect(self._cancel_button_clicked)
        self.buttons_layout.addWidget(self.cancel_button)

        self.create_document_base_button = QPushButton("Create Document Base")
        self.create_document_base_button.setFont(BUTTON_FONT)
        self.create_document_base_button.clicked.connect(self._create_document_base_button_clicked)
        self.buttons_layout.addWidget(self.create_document_base_button)

        self._create_attribute_button_clicked(None)

    def initialize_for_new_document_base(self):

        # remove all the attribute widgets
        while 0 < self.num_visible_create_attribute_widgets:
            widget = self.create_attribute_widgets[self.num_visible_create_attribute_widgets - 1]
            widget.hide()
            self.create_attribute_list_layout.removeWidget(widget)
            self.num_visible_create_attribute_widgets -= 1

    def _edit_path_button_clicked(self, _):
        path = str(QFileDialog.getExistingDirectory(self, "Choose a directory of text files."))
        if path != "":
            path = f"{path}/*.txt"
            self.path.setText(path)

    def _create_attribute_button_clicked(self, _):

        # make sure there are enough create attribute widgets
        if len(self.create_attribute_widgets) < self.num_visible_create_attribute_widgets + 1:
            self.create_attribute_widgets.append(CreateAttributeWidget(self))

        # initialize the new create attribute widget and show it
        widget = self.create_attribute_widgets[self.num_visible_create_attribute_widgets]
        widget.initialize_for_new_attribute()
        self.create_attribute_list_layout.addWidget(widget)
        widget.show()
        self.num_visible_create_attribute_widgets += 1
        self.create_attribute_list_layout.removeWidget(self.create_attribute_button)
        self.create_attribute_list_layout.addWidget(self.create_attribute_button)

    def _cancel_button_clicked(self, _):
        self.main_window.show_document_base_viewer_widget()

    def _create_document_base_button_clicked(self, _):
        path = self.path.text()
        attribute_names = []
        for widget in self.create_attribute_widgets[:self.num_visible_create_attribute_widgets]:
            attribute_names.append(widget.name.text())
        self.main_window.create_new_document_base(path, attribute_names)

    def delete_attribute(self, attribute_widget):
        attribute_widget.hide()
        self.create_attribute_list_layout.removeWidget(attribute_widget)
        self.create_attribute_widgets.remove(attribute_widget)
        self.create_attribute_widgets.append(attribute_widget)
        self.num_visible_create_attribute_widgets -= 1
        self.create_attribute_list_layout.removeWidget(self.create_attribute_button)
        self.create_attribute_list_layout.addWidget(self.create_attribute_button)


class CreateAttributeWidget(QFrame):
    def __init__(self, create_document_base_widget) -> None:
        super(CreateAttributeWidget, self).__init__()

        self.create_document_base_widget = create_document_base_widget

        self.setFixedHeight(40)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        self.layout.setSpacing(10)
        self.setStyleSheet("background-color: white")

        self.name = QLineEdit()
        self.name.setFont(CODE_FONT_BOLD)
        self.name.setStyleSheet("border: none")
        self.layout.addWidget(self.name)

        self.delete_button = QPushButton()
        self.delete_button.setIcon(QIcon("aset_ui/resources/incorrect.svg"))
        self.delete_button.setFlat(True)
        self.delete_button.clicked.connect(self._delete_button_clicked)
        self.layout.addWidget(self.delete_button)

    def initialize_for_new_attribute(self):
        self.name.setText("")

    def _delete_button_clicked(self, _):
        self.create_document_base_widget.delete_attribute(self)
