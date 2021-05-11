import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QScrollArea, QPushButton, QFrame, QLineEdit

from aset_ui.util import SUBHEADER_FONT, LABEL_FONT, LABEL_FONT_ITALIC

logger = logging.getLogger(__name__)


class AttributesInputWidget(QWidget):
    """Widget to enter the attributes and provide examples for them."""

    def __init__(self, parent):
        super(AttributesInputWidget, self).__init__(parent)
        self._parent = parent

        # layout and header
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(self._layout)

        self._subheader = QLabel("3. Enter attributes to extract from the documents.")
        self._subheader.setFont(SUBHEADER_FONT)
        self._layout.addWidget(self._subheader)

        self._label = QLabel("Each attribute must have a unique name. You may also provide example values for each "
                             "attribute.")
        self._label.setWordWrap(True)
        self._label.setFont(LABEL_FONT)
        self._layout.addWidget(self._label)

        # list of attributes
        self._attribute_widgets = []

        self._list_widget = QWidget()
        self._list_layout = QHBoxLayout()
        self._list_widget.setLayout(self._list_layout)
        self._list_layout.setContentsMargins(10, 10, 10, 10)
        self._list_layout.setAlignment(Qt.AlignTop)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._list_widget)
        self._layout.addWidget(self._scroll_area)

        # add attribute button
        self._add_attribute_button = QPushButton("Add Attribute")
        self._add_attribute_button.clicked.connect(self.add_attribute)
        self._add_attribute_button.setFixedWidth(100)
        self._list_layout.addWidget(self._add_attribute_button)

        # feedback label
        self._feedback_label = QLabel(" ")
        self._feedback_label.setFont(LABEL_FONT_ITALIC)
        self._layout.addWidget(self._feedback_label)

    def add_attribute(self):
        new_attribute_widget = AttributeWidget(self)
        self._attribute_widgets.append(new_attribute_widget)
        self._list_layout.addWidget(new_attribute_widget)
        self._list_layout.removeWidget(self._add_attribute_button)
        self._list_layout.addWidget(self._add_attribute_button)

        self.attributes_changed()

    def remove_attribute(self, attribute_widget):
        attribute_widget.hide()
        self._list_layout.removeWidget(attribute_widget)
        self._attribute_widgets.remove(attribute_widget)
        attribute_widget.deleteLater()

        self.attributes_changed()

    def attributes_changed(self):
        self._scroll_area.update()

        self._feedback_label.setStyleSheet("color: black")
        self._feedback_label.setText(" ")

    def give_feedback(self, feedback):
        self._feedback_label.setStyleSheet("color: red")
        self._feedback_label.setText(feedback)

    def get_attribute_widgets(self):
        return self._attribute_widgets


class AttributeWidget(QFrame):
    """Widget to display a single attribute."""

    def __init__(self, parent):
        super(AttributeWidget, self).__init__(parent)
        self._parent = parent

        # layout
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setFixedWidth(300)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setAlignment(Qt.AlignTop)

        # top widget: attribute name and remove attribute button
        self._top_widget = QWidget()
        self._top_layout = QHBoxLayout()
        self._top_widget.setLayout(self._top_layout)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._top_widget)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("attribute name")
        self._name_edit.textChanged.connect(self._parent.attributes_changed)
        self._top_layout.addWidget(self._name_edit)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(lambda: self._parent.remove_attribute(self))
        self._top_layout.addWidget(self._remove_button)

        # list of examples
        self._example_widgets = []

        self._examples_widget = QWidget()
        self._examples_layout = QVBoxLayout()
        self._examples_widget.setLayout(self._examples_layout)
        self._examples_layout.setContentsMargins(30, 0, 0, 0)
        self._layout.addWidget(self._examples_widget)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout()
        self._list_widget.setLayout(self._list_layout)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._examples_layout.addWidget(self._list_widget)

        self._add_example_button = QPushButton("Add Example Value")
        self._add_example_button.clicked.connect(self.add_example)
        self._examples_layout.addWidget(self._add_example_button)

    def add_example(self):
        new_example_widget = ExampleWidget(self)
        self._example_widgets.append(new_example_widget)
        self._list_layout.addWidget(new_example_widget)

        self.examples_changed()

    def remove_example(self, example_widget):
        example_widget.hide()
        self._list_layout.removeWidget(example_widget)
        self._example_widgets.remove(example_widget)
        example_widget.deleteLater()

        self.examples_changed()

    def examples_changed(self):
        self._parent.attributes_changed()


class ExampleWidget(QWidget):
    """Widget to display a single example."""

    def __init__(self, parent):
        super(ExampleWidget, self).__init__(parent)
        self._parent = parent

        # layout
        self._layout = QHBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # value and remove button
        self._value_edit = QLineEdit()
        self._value_edit.setPlaceholderText("example value")
        self._value_edit.textChanged.connect(self._parent.examples_changed)
        self._layout.addWidget(self._value_edit)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(lambda: self._parent.remove_example(self))
        self._layout.addWidget(self._remove_button)
