import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QPushButton, QScrollArea

from aset.data.signals import CachedContextSentenceSignal
from aset_ui.style import HEADER_FONT, CODE_FONT

logger = logging.getLogger(__name__)


class InteractiveMatchingWidget(QWidget):

    def __init__(self, main_window):
        super(InteractiveMatchingWidget, self).__init__(main_window)
        self._main_window = main_window

        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        # title
        self._header = QLabel("Matching Attribute '':")
        self._header.setFont(HEADER_FONT)
        self._layout.addWidget(self._header)

        # content
        self._nugget_list_widget = NuggetListWidget(self)
        self._document_widget = DocumentWidget(self)
        self._show_nugget_list_widget()

        logger.debug("Initialized InteractiveMatchingWidget.")

    def handle_feedback_request(self, feedback_request):
        logger.info("Handle feedback request.")
        self._show_nugget_list_widget()
        self._header.setText(f"Matching Attribute '{feedback_request['attribute'].name}':")
        self._nugget_list_widget.update_nuggets(feedback_request["nuggets"])

    def give_feedback(self, feedback):
        logger.info("Give feedback.")
        self._main_window.give_feedback(feedback)

    def get_document_feedback(self, nugget):
        logger.info("Get document feedback.")
        self._show_document_widget()
        self._document_widget.update_document(nugget)

    def _show_nugget_list_widget(self):
        self._document_widget.hide()
        self._nugget_list_widget.show()
        self._layout.removeWidget(self._document_widget)
        self._layout.addWidget(self._nugget_list_widget)

    def _show_document_widget(self):
        self._nugget_list_widget.hide()
        self._document_widget.show()
        self._layout.removeWidget(self._nugget_list_widget)
        self._layout.addWidget(self._document_widget)

    def enable_input(self):
        pass

    def disable_input(self):
        pass


class NuggetListWidget(QWidget):

    def __init__(self, interactive_matching_widget):
        super(NuggetListWidget, self).__init__(interactive_matching_widget)
        self._interactive_matching_widget = interactive_matching_widget

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._nugget_list_item_widgets = []
        self._vertical_list = QWidget()
        self._vertical_list_layout = QVBoxLayout()
        self._vertical_list_layout.setContentsMargins(10, 10, 10, 10)
        self._vertical_list.setLayout(self._vertical_list_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(0)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._vertical_list)
        layout.addWidget(scroll_area)

        stop_button = QPushButton("Continue With Next Attribute")
        stop_button.clicked.connect(self._stop_button_clicked)
        layout.addWidget(stop_button)

        logger.debug("Initialized NuggetListWidget.")

    def update_nuggets(self, nuggets):
        for widget in reversed(self._nugget_list_item_widgets):
            self._nugget_list_item_widgets.remove(widget)
            self._vertical_list_layout.removeWidget(widget)
            widget.deleteLater()
            widget.setParent(None)
        max_len = max(len(nugget.signals[CachedContextSentenceSignal.signal_str].value["text"]) for nugget in nuggets)
        self._nugget_list_item_widgets = []
        for nugget in nuggets:
            widget = NuggetListItemWidget(self, nugget, max_len)
            self._vertical_list_layout.addWidget(widget)
            self._nugget_list_item_widgets.append(widget)

    def give_feedback(self, feedback):
        self._interactive_matching_widget.give_feedback(feedback)

    def get_document_feedback(self, nugget):
        self._interactive_matching_widget.get_document_feedback(nugget)

    def _stop_button_clicked(self, _):
        self._interactive_matching_widget.give_feedback({
            "message": "stop-interactive-matching"
        })


class NuggetListItemWidget(QWidget):

    def __init__(self, nugget_list_widget, nugget, padding_left):
        super(NuggetListItemWidget, self).__init__(nugget_list_widget)
        self._nugget_list_widget = nugget_list_widget
        self._nugget = nugget

        layout = QHBoxLayout()
        self.setLayout(layout)

        text_edit = QTextEdit()
        layout.addWidget(text_edit)
        text_edit.setReadOnly(True)
        text_edit.setFont(CODE_FONT)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
        text_edit.setLineWrapColumnOrWidth(10000)
        text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setFixedHeight(33)
        text_edit.setText("")

        context = nugget.signals[CachedContextSentenceSignal.signal_str].value
        sentence = context["text"]
        start_char = context["start_char"]
        end_char = context["end_char"]

        formatted_text = f"{'&#160;' * padding_left}{sentence[:start_char]}" \
                         f"<span style='background-color: #FFFF00'><b>{sentence[start_char:end_char]}</b></span>" \
                         f"{sentence[end_char:]}"
        text_edit.textCursor().insertHtml(formatted_text)

        scroll_cursor = QTextCursor(text_edit.document())
        scroll_cursor.setPosition(padding_left + start_char)
        text_edit.setTextCursor(scroll_cursor)
        text_edit.ensureCursorVisible()

        match_button = QPushButton("MATCH")
        match_button.clicked.connect(self._match_button_clicked)
        layout.addWidget(match_button)

        correct_button = QPushButton("CORRECT")
        correct_button.clicked.connect(self._correct_button_clicked)
        layout.addWidget(correct_button)

    def _match_button_clicked(self, _):
        self._nugget_list_widget.give_feedback({
            "message": "is-match",
            "nugget": self._nugget
        })

    def _correct_button_clicked(self, _):
        self._nugget_list_widget.get_document_feedback(self._nugget)


class DocumentWidget(QWidget):

    def __init__(self, interactive_matching_widget):
        super(DocumentWidget, self).__init__(interactive_matching_widget)
        self._interactive_matching_widget = interactive_matching_widget

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._document = None
        self._current_nugget = None
        self._base_formatted_text = ""
        self._idx_mapper = {}
        self._nuggets_in_order = []

        self._text_edit = QTextEdit()
        layout.addWidget(self._text_edit)
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(CODE_FONT)
        self._text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._text_edit.setText("")

        buttons_widget = QWidget()
        layout.addWidget(buttons_widget)
        buttons_widget_layout = QHBoxLayout()
        buttons_widget.setLayout(buttons_widget_layout)

        left_button = QPushButton("<")
        left_button.clicked.connect(self._left_button_clicked)
        buttons_widget_layout.addWidget(left_button)

        right_button = QPushButton(">")
        right_button.clicked.connect(self._right_button_clicked)
        buttons_widget_layout.addWidget(right_button)

        match_button = QPushButton("MATCH")
        match_button.clicked.connect(self._match_button_clicked)
        buttons_widget_layout.addWidget(match_button)

        no_match_button = QPushButton("NO MATCH IN THIS DOCUMENT")
        no_match_button.clicked.connect(self._no_match_button_clicked)
        buttons_widget_layout.addWidget(no_match_button)

        logger.debug("Initialized DocumentWidget.")

    def _left_button_clicked(self, _):
        idx = self._nuggets_in_order.index(self._current_nugget)
        if idx > 0:
            self._current_nugget = self._nuggets_in_order[idx - 1]
            self._highlight_current_nugget()

    def _right_button_clicked(self, _):
        idx = self._nuggets_in_order.index(self._current_nugget)
        if idx < len(self._nuggets_in_order) - 1:
            self._current_nugget = self._nuggets_in_order[idx + 1]
            self._highlight_current_nugget()

    def _match_button_clicked(self, _):
        self._interactive_matching_widget.give_feedback({
            "message": "is-match",
            "nugget": self._current_nugget
        })

    def _no_match_button_clicked(self, _):
        self._interactive_matching_widget.give_feedback({
            "message": "no-match-in-document",
            "nugget": self._current_nugget
        })

    def _highlight_current_nugget(self):
        mapped_start_char = self._idx_mapper[self._current_nugget.start_char]
        mapped_end_char = self._idx_mapper[self._current_nugget.end_char]
        formatted_text = f"{self._base_formatted_text[:mapped_start_char]}" \
                         f"<span style='background-color: #FFFF00'><b>{self._base_formatted_text[mapped_start_char:mapped_end_char]}</span></b>" \
                         f"{self._base_formatted_text[mapped_end_char:]}"
        self._text_edit.setText("")
        self._text_edit.textCursor().insertHtml(formatted_text)

    def update_document(self, nugget):
        self._document = nugget.document
        self._current_nugget = nugget
        self._nuggets_in_order = list(sorted(self._document.nuggets, key=lambda x: x.start_char))

        if self._nuggets_in_order != []:
            self._idx_mapper = {}
            char_list = []
            end_chars = []
            next_start_char = 0
            next_nugget_idx = 0
            for idx, char in enumerate(list(self._document.text)):
                if idx == next_start_char:
                    if end_chars == []:
                        char_list += list("<b>")
                    end_chars.append(self._nuggets_in_order[next_nugget_idx].end_char)
                    next_nugget_idx += 1
                    if next_nugget_idx < len(self._nuggets_in_order):
                        next_start_char = self._nuggets_in_order[next_nugget_idx].start_char
                    else:
                        next_start_char = -1
                while idx in end_chars:
                    end_chars.remove(idx)
                if end_chars == []:
                    char_list += list("</b>")
                self._idx_mapper[idx] = len(char_list)
                char_list.append(char)
            self._base_formatted_text = "".join(char_list)
        else:
            self._idx_mapper = {}
            for idx in range(len(self._document.text)):
                self._idx_mapper[idx] = idx
            self._base_formatted_text = ""

        self._highlight_current_nugget()

        scroll_cursor = QTextCursor(self._text_edit.document())
        scroll_cursor.setPosition(nugget.start_char)
        self._text_edit.setTextCursor(scroll_cursor)
        self._text_edit.ensureCursorVisible()
