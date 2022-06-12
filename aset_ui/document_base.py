from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from aset_ui.common import BUTTON_FONT, CODE_FONT, CODE_FONT_BOLD, LABEL_FONT, MainWindowContent, \
    MainWindowContentSection, CustomScrollableListItem, CustomScrollableList


class DocumentBaseViewerWidget(MainWindowContent):
    def __init__(self, main_window):
        super(DocumentBaseViewerWidget, self).__init__(main_window, "Document Base")

        # controls
        self.controls = QWidget()
        self.controls_layout = QHBoxLayout(self.controls)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(10)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.addWidget(self.controls)

        self.create_document_base_button = QPushButton("Create a new Document Base")
        self.create_document_base_button.setFont(BUTTON_FONT)
        self.create_document_base_button.clicked.connect(self.main_window.show_document_base_creator_widget_task)
        self.controls_layout.addWidget(self.create_document_base_button)

        self.load_and_run_default_preprocessing_phase_button = QPushButton("Preprocess the Document Base")
        self.load_and_run_default_preprocessing_phase_button.setFont(BUTTON_FONT)
        self.load_and_run_default_preprocessing_phase_button.clicked.connect(
            self.main_window.load_and_run_default_preprocessing_phase_task
        )
        self.controls_layout.addWidget(self.load_and_run_default_preprocessing_phase_button)

        self.enter_query_button = QPushButton("Enter Query to Derive Attributes")
        self.enter_query_button.setFont(BUTTON_FONT)
        self.enter_query_button.clicked.connect(self.main_window.enter_query_task)
        self.controls_layout.addWidget(self.enter_query_button)

        self.load_and_run_default_matching_phase_button = QPushButton("Match the Nuggets to the Attributes")
        self.load_and_run_default_matching_phase_button.setFont(BUTTON_FONT)
        self.load_and_run_default_matching_phase_button.clicked.connect(
            self.main_window.load_and_run_default_matching_phase_task
        )
        self.controls_layout.addWidget(self.load_and_run_default_matching_phase_button)

        self.save_table_button = QPushButton("Export the Table to CSV")
        self.save_table_button.setFont(BUTTON_FONT)
        self.save_table_button.clicked.connect(self.main_window.save_table_to_csv_task)
        self.controls_layout.addWidget(self.save_table_button)

        # documents
        self.documents = MainWindowContentSection(self, "Documents:")
        self.layout.addWidget(self.documents)

        self.num_documents = QLabel("number of documents: -")
        self.num_documents.setFont(LABEL_FONT)
        self.documents.layout.addWidget(self.num_documents)

        self.num_nuggets = QLabel("number of nuggets: -")
        self.num_nuggets.setFont(LABEL_FONT)
        self.documents.layout.addWidget(self.num_nuggets)

        # attributes
        self.attributes = MainWindowContentSection(self, "Attributes:")
        self.layout.addWidget(self.attributes)

        self.add_attribute_button = QPushButton("Add Attribute")
        self.add_attribute_button.setFont(BUTTON_FONT)
        self.add_attribute_button.clicked.connect(self.main_window.add_attribute_task)

        self.attributes_list = CustomScrollableList(self, AttributeWidget, self.add_attribute_button)
        self.attributes.layout.addWidget(self.attributes_list)

    def update_document_base(self, document_base):
        # update documents
        self.num_documents.setText(f"number of documents: {len(document_base.documents)}")
        self.num_nuggets.setText(f"number of nuggets: {len(document_base.nuggets)}")

        # update attributes
        self.attributes_list.update_item_list(document_base.attributes, document_base)

    def enable_input(self):
        self.create_document_base_button.setEnabled(True)

        if self.main_window.document_base is not None:
            self.load_and_run_default_preprocessing_phase_button.setEnabled(True)
            self.enter_query_button.setEnabled(True)
            self.load_and_run_default_matching_phase_button.setEnabled(True)
            self.save_table_button.setEnabled(True)
            self.add_attribute_button.setEnabled(True)
            self.attributes_list.enable_input()

    def disable_input(self):
        self.create_document_base_button.setDisabled(True)
        self.load_and_run_default_preprocessing_phase_button.setDisabled(True)
        self.enter_query_button.setDisabled(True)
        self.load_and_run_default_matching_phase_button.setDisabled(True)
        self.save_table_button.setDisabled(True)
        self.add_attribute_button.setDisabled(True)
        self.attributes_list.disable_input()


class AttributeWidget(CustomScrollableListItem):
    def __init__(self, document_base_viewer):
        super(AttributeWidget, self).__init__(document_base_viewer)
        self.document_base_viewer = document_base_viewer
        self.attribute = None

        self.setFixedHeight(40)
        self.setStyleSheet("background-color: white")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 20, 0)
        self.layout.setSpacing(40)

        self.attribute_name = QLabel()
        self.attribute_name.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.attribute_name, alignment=Qt.AlignmentFlag.AlignLeft)

        self.num_matched = QLabel("matches: -")
        self.num_matched.setFont(CODE_FONT)
        self.layout.addWidget(self.num_matched, alignment=Qt.AlignmentFlag.AlignLeft)

        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setSpacing(10)
        self.layout.addWidget(self.buttons_widget, alignment=Qt.AlignmentFlag.AlignRight)

        self.forget_matches_button = QPushButton()
        self.forget_matches_button.setIcon(QIcon("aset_ui/resources/redo.svg"))
        self.forget_matches_button.setToolTip("Forget matches for this attribute.")
        self.forget_matches_button.setFlat(True)
        self.forget_matches_button.clicked.connect(self._forget_matches_button_clicked)
        self.buttons_layout.addWidget(self.forget_matches_button)

        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon("aset_ui/resources/trash.svg"))
        self.remove_button.setToolTip("Remove this attribute.")
        self.remove_button.setFlat(True)
        self.remove_button.clicked.connect(self._remove_button_clicked)
        self.buttons_layout.addWidget(self.remove_button)

    def update_item(self, item, params=None):
        self.attribute = item

        if len(params.attributes) == 0:
            max_attribute_name_len = 10
        else:
            max_attribute_name_len = max(len(attribute.name) for attribute in params.attributes)
        self.attribute_name.setText(self.attribute.name + (" " * (max_attribute_name_len - len(self.attribute.name))))

        mappings_in_some_documents = False
        no_mappings_in_some_documents = False
        num_matches = 0
        for document in params.documents:
            if self.attribute.name in document.attribute_mappings.keys():
                mappings_in_some_documents = True
                if document.attribute_mappings[self.attribute.name] != []:
                    num_matches += 1
            else:
                no_mappings_in_some_documents = True

        if not mappings_in_some_documents and no_mappings_in_some_documents:
            self.num_matched.setText("not matched yet")
        elif mappings_in_some_documents and no_mappings_in_some_documents:
            self.num_matched.setText("only partly matched")
        else:
            self.num_matched.setText(f"matches: {num_matches}")

    def enable_input(self):
        self.forget_matches_button.setEnabled(True)
        self.remove_button.setEnabled(True)

    def disable_input(self):
        self.forget_matches_button.setDisabled(True)
        self.remove_button.setDisabled(True)

    def _forget_matches_button_clicked(self):
        self.document_base_viewer.main_window.forget_matches_for_attribute_with_given_name_task(self.attribute.name)

    def _remove_button_clicked(self):
        self.document_base_viewer.main_window.remove_attribute_with_given_name_task(self.attribute.name)


class DocumentBaseCreatorWidget(MainWindowContent):
    def __init__(self, main_window) -> None:
        super(DocumentBaseCreatorWidget, self).__init__(main_window, "Create Document Base")

        self.documents = MainWindowContentSection(self, "Documents:")
        self.layout.addWidget(self.documents)

        self.documents_explanation = QLabel(
            "Enter the path of the directory that contains the documents as .txt files."
        )
        self.documents_explanation.setFont(LABEL_FONT)
        self.documents.layout.addWidget(self.documents_explanation)

        self.path_widget = QFrame()
        self.path_layout = QHBoxLayout(self.path_widget)
        self.path_layout.setContentsMargins(20, 0, 20, 0)
        self.path_layout.setSpacing(10)
        self.path_widget.setStyleSheet("background-color: white")
        self.path_widget.setFixedHeight(40)
        self.documents.layout.addWidget(self.path_widget)

        self.path = QLineEdit()
        self.path.setFont(CODE_FONT_BOLD)
        self.path.setStyleSheet("border: none")
        self.path_layout.addWidget(self.path)

        self.edit_path_button = QPushButton()
        self.edit_path_button.setIcon(QIcon("aset_ui/resources/folder.svg"))
        self.edit_path_button.setFlat(True)
        self.edit_path_button.clicked.connect(self._edit_path_button_clicked)
        self.path_layout.addWidget(self.edit_path_button)

        self.attributes = MainWindowContentSection(self, "Attributes:")
        self.layout.addWidget(self.attributes)

        self.labels_explanation = QLabel("Enter the attribute names.")
        self.labels_explanation.setFont(LABEL_FONT)
        self.attributes.layout.addWidget(self.labels_explanation)

        self.create_attribute_button = QPushButton("New Attribute")
        self.create_attribute_button.setFont(BUTTON_FONT)
        self.create_attribute_button.clicked.connect(self._create_attribute_button_clicked)

        self.attribute_names = []
        self.attributes_list = CustomScrollableList(self, AttributeCreatorWidget, self.create_attribute_button)
        self.attributes.layout.addWidget(self.attributes_list)

        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setSpacing(10)
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.attributes.layout.addWidget(self.buttons_widget)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(BUTTON_FONT)
        self.cancel_button.clicked.connect(self._cancel_button_clicked)
        self.buttons_layout.addWidget(self.cancel_button)

        self.create_document_base_button = QPushButton("Create Document Base")
        self.create_document_base_button.setFont(BUTTON_FONT)
        self.create_document_base_button.clicked.connect(self._create_document_base_button_clicked)
        self.buttons_layout.addWidget(self.create_document_base_button)

    def enable_input(self):
        self.edit_path_button.setEnabled(True)
        self.create_attribute_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.create_document_base_button.setEnabled(True)
        self.attributes_list.enable_input()

    def disable_input(self):
        self.edit_path_button.setDisabled(True)
        self.create_attribute_button.setDisabled(True)
        self.cancel_button.setDisabled(True)
        self.create_document_base_button.setDisabled(True)
        self.attributes_list.disable_input()

    def initialize_for_new_document_base(self):
        self.path.setText("")
        self.attribute_names = []
        self.attributes_list.update_item_list([])

    def delete_attribute(self, attribute_name):
        self.attribute_names = []
        for attribute_widget in self.attributes_list.item_widgets[:self.attributes_list.num_visible_item_widgets]:
            self.attribute_names.append(attribute_widget.name.text())
        self.attribute_names.remove(attribute_name)
        self.attributes_list.update_item_list(self.attribute_names)
        self.attributes_list.last_item_widget().name.setFocus()

    def _edit_path_button_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Choose a directory of text files."))
        if path != "":
            path = f"{path}/*.txt"
            self.path.setText(path)

    def _create_attribute_button_clicked(self):
        self.attribute_names = []
        for attribute_widget in self.attributes_list.item_widgets[:self.attributes_list.num_visible_item_widgets]:
            self.attribute_names.append(attribute_widget.name.text())
        self.attribute_names.append("")
        self.attributes_list.update_item_list(self.attribute_names)
        self.attributes_list.last_item_widget().name.setFocus()

    def _cancel_button_clicked(self):
        self.main_window.show_start_menu_widget()
        self.main_window.enable_global_input()

    def _create_document_base_button_clicked(self):
        self.attribute_names = []
        for attribute_widget in self.attributes_list.item_widgets[:self.attributes_list.num_visible_item_widgets]:
            self.attribute_names.append(attribute_widget.name.text())
        self.main_window.create_document_base_task(self.path.text(), self.attribute_names)


class AttributeCreatorWidget(CustomScrollableListItem):
    def __init__(self, document_base_creator_widget) -> None:
        super(AttributeCreatorWidget, self).__init__(document_base_creator_widget)
        self.document_base_creator_widget = document_base_creator_widget

        self.setFixedHeight(40)
        self.setStyleSheet("background-color: white")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 20, 0)
        self.layout.setSpacing(10)

        self.name = QLineEdit()
        self.name.setFont(CODE_FONT_BOLD)
        self.name.setStyleSheet("border: none")
        self.layout.addWidget(self.name)

        self.delete_button = QPushButton()
        self.delete_button.setIcon(QIcon("aset_ui/resources/trash.svg"))
        self.delete_button.setFlat(True)
        self.delete_button.clicked.connect(self._delete_button_clicked)
        self.layout.addWidget(self.delete_button)

    def update_item(self, item, params=None):
        self.name.setText(item)

    def _delete_button_clicked(self):
        self.document_base_creator_widget.delete_attribute(self.name.text())

    def enable_input(self):
        self.delete_button.setEnabled(True)

    def disable_input(self):
        self.delete_button.setEnabled(False)
