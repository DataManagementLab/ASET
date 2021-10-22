from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QScrollArea, QFrame

from aset_ui.style import HEADER_FONT, SUBHEADER_FONT, LABEL_FONT, CODE_FONT, CODE_FONT_BOLD


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
        for attribute, widget in zip(document_base.attributes, self.attribute_widgets[:len(document_base.attributes)]):
            widget.update_attribute(attribute, document_base, max_attribute_name_len)

    def enable_input(self):
        pass

    def disable_input(self):
        pass


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
            if attribute.name in document.attribute_mappings.keys() \
                    and document.attribute_mappings[attribute.name] != []:
                num_matches += 1
        self.num_matched.setText(f"matches: {num_matches}")
