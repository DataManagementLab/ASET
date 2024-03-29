import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QMessageBox

from aset.data.signals import CachedContextSentenceSignal, CachedDistanceSignal
from aset_ui.common import BUTTON_FONT, CODE_FONT, CODE_FONT_BOLD, LABEL_FONT, MainWindowContent, \
    CustomScrollableList, CustomScrollableListItem, WHITE, LIGHT_RED, LIGHT_YELLOW, YELLOW

logger = logging.getLogger(__name__)


class InteractiveMatchingWidget(MainWindowContent):
    def __init__(self, main_window):
        super(InteractiveMatchingWidget, self).__init__(main_window, "Matching Attribute:")

        self.nugget_list_widget = NuggetListWidget(self)
        self.document_widget = DocumentWidget(self)

        self.show_nugget_list_widget()

    def enable_input(self):
        self.nugget_list_widget.enable_input()
        self.document_widget.enable_input()

    def disable_input(self):
        self.nugget_list_widget.disable_input()
        self.document_widget.disable_input()

    def handle_feedback_request(self, feedback_request):
        self.header.setText(f"Matching Attribute '{feedback_request['attribute'].name}':")
        self.nugget_list_widget.update_nuggets(feedback_request)
        self.show_nugget_list_widget()

    def get_document_feedback(self, nugget):
        self.document_widget.update_document(nugget)
        self.show_document_widget()

    def show_nugget_list_widget(self):
        self.document_widget.hide()
        self.nugget_list_widget.show()
        self.layout.removeWidget(self.document_widget)
        self.layout.addWidget(self.nugget_list_widget)

    def show_document_widget(self):
        self.nugget_list_widget.hide()
        self.document_widget.show()
        self.layout.removeWidget(self.nugget_list_widget)
        self.layout.addWidget(self.document_widget)


class NuggetListWidget(QWidget):
    def __init__(self, interactive_matching_widget):
        super(NuggetListWidget, self).__init__(interactive_matching_widget)
        self.interactive_matching_widget = interactive_matching_widget

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        # top widget
        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(10)
        self.layout.addWidget(self.top_widget)

        self.description = QLabel("Below you see a list of guessed matches for you to confirm or fix.")
        self.description.setFont(LABEL_FONT)
        self.top_layout.addWidget(self.description)

        self.stop_button = QPushButton("Continue With Next Attribute")
        self.stop_button.setFont(BUTTON_FONT)
        self.stop_button.clicked.connect(self._stop_button_clicked)
        self.stop_button.setMaximumWidth(240)
        self.top_layout.addWidget(self.stop_button)

        # nugget list
        self.nugget_list = CustomScrollableList(self, NuggetListItemWidget)
        self.layout.addWidget(self.nugget_list)

    def update_nuggets(self, feedback_request):
        nuggets = feedback_request["nuggets"]
        params = {
            "max_start_chars": max([nugget[CachedContextSentenceSignal]["start_char"] for nugget in nuggets]),
            "max_distance": feedback_request["max-distance"]
        }
        self.nugget_list.update_item_list(nuggets, params)

    def _stop_button_clicked(self):
        self.interactive_matching_widget.main_window.give_feedback_task({"message": "stop-interactive-matching"})

    def enable_input(self):
        self.stop_button.setEnabled(True)
        self.nugget_list.enable_input()

    def disable_input(self):
        self.stop_button.setDisabled(True)
        self.nugget_list.disable_input()


class NuggetListItemWidget(CustomScrollableListItem):
    def __init__(self, nugget_list_widget):
        super(NuggetListItemWidget, self).__init__(nugget_list_widget)
        self.nugget_list_widget = nugget_list_widget
        self.nugget = None

        self.setFixedHeight(45)
        self.setStyleSheet(f"background-color: {WHITE}")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 20, 0)
        self.layout.setSpacing(10)

        self.info_button = QPushButton()
        self.info_button.setFlat(True)
        self.info_button.setFont(CODE_FONT_BOLD)
        self.info_button.clicked.connect(self._info_button_clicked)
        self.layout.addWidget(self.info_button)

        self.left_split_label = QLabel("|")
        self.left_split_label.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.left_split_label)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFrameStyle(0)
        self.text_edit.setFont(CODE_FONT)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
        self.text_edit.setLineWrapColumnOrWidth(10000)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setFixedHeight(27)
        self.text_edit.setText("")
        self.layout.addWidget(self.text_edit)

        self.right_split_label = QLabel("|")
        self.right_split_label.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.right_split_label)

        self.match_button = QPushButton()
        self.match_button.setIcon(QIcon("aset_ui/resources/correct.svg"))
        self.match_button.setFlat(True)
        self.match_button.clicked.connect(self._match_button_clicked)
        self.layout.addWidget(self.match_button)

        self.fix_button = QPushButton()
        self.fix_button.setIcon(QIcon("aset_ui/resources/magnifier.svg"))
        self.fix_button.setFlat(True)
        self.fix_button.clicked.connect(self._fix_button_clicked)
        self.layout.addWidget(self.fix_button)

    def update_item(self, item, params=None):
        self.nugget = item

        max_start_chars = params["max_start_chars"]
        max_distance = params["max_distance"]

        if max_distance < self.nugget[CachedDistanceSignal]:
            self.setStyleSheet(f"background-color: {LIGHT_RED}")
        else:
            self.setStyleSheet(f"background-color: {WHITE}")

        sentence = self.nugget[CachedContextSentenceSignal]["text"]
        start_char = self.nugget[CachedContextSentenceSignal]["start_char"]
        end_char = self.nugget[CachedContextSentenceSignal]["end_char"]

        self.text_edit.setText("")
        formatted_text = (
            f"{'&#160;' * (max_start_chars - start_char)}{sentence[:start_char]}"
            f"<span style='background-color: {YELLOW}'><b>{sentence[start_char:end_char]}</b></span>"
            f"{sentence[end_char:]}{'&#160;' * 50}"
        )
        self.text_edit.textCursor().insertHtml(formatted_text)

        scroll_cursor = QTextCursor(self.text_edit.document())
        scroll_cursor.setPosition(max_start_chars + 50)
        self.text_edit.setTextCursor(scroll_cursor)
        self.text_edit.ensureCursorVisible()

        self.info_button.setText(f"{str(round(self.nugget[CachedDistanceSignal], 2)).ljust(4)}")

    def _match_button_clicked(self):
        self.nugget_list_widget.interactive_matching_widget.main_window.give_feedback_task({
            "message": "is-match",
            "nugget": self.nugget
        })

    def _fix_button_clicked(self):
        self.nugget_list_widget.interactive_matching_widget.get_document_feedback(self.nugget)

    def _info_button_clicked(self):
        lines = []
        lines.append("Signal values:")
        lines.append("")
        for key, value in self.nugget.signals.items():
            lines.append(f"- {key}: '{str(value)[:40]}'")

        lines.append("")
        lines.append("All nuggets in document:")
        lines.append("")
        nuggets = self.nugget.document.nuggets
        nuggets = list(sorted(nuggets, key=lambda x: x[CachedDistanceSignal]))
        for nugget in nuggets:
            lines.append(f"- '{nugget.text}' ({nugget[CachedDistanceSignal]})")

        QMessageBox.information(self, "Nugget Information", "\n".join(lines))

    def enable_input(self):
        self.match_button.setEnabled(True)
        self.fix_button.setEnabled(True)

    def disable_input(self):
        self.match_button.setDisabled(True)
        self.fix_button.setDisabled(True)


class DocumentWidget(QWidget):
    def __init__(self, interactive_matching_widget):
        super(DocumentWidget, self).__init__(interactive_matching_widget)
        self.interactive_matching_widget = interactive_matching_widget

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        self.setLayout(self.layout)

        self.document = None
        self.current_nugget = None
        self.base_formatted_text = ""
        self.idx_mapper = {}
        self.nuggets_in_order = []
        self.nuggets_sorted_by_distance = []

        self.text_edit = QTextEdit()
        self.layout.addWidget(self.text_edit)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFrameStyle(0)
        self.text_edit.setFont(CODE_FONT)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setText("")

        self.suggestion_list = CustomScrollableList(self, SuggestionListItemWidget, orientation="horizontal")
        self.suggestion_list.setFixedHeight(60)
        self.layout.addWidget(self.suggestion_list)

        self.buttons_widget = QWidget()
        self.buttons_widget_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.buttons_widget)

        self.left_button = QPushButton("Skip Left")
        self.left_button.setFont(BUTTON_FONT)
        self.left_button.clicked.connect(self._left_button_clicked)
        self.buttons_widget_layout.addWidget(self.left_button)

        self.right_button = QPushButton("Skip Right")
        self.right_button.setFont(BUTTON_FONT)
        self.right_button.clicked.connect(self._right_button_clicked)
        self.buttons_widget_layout.addWidget(self.right_button)

        self.match_button = QPushButton("Confirm Match")
        self.match_button.setFont(BUTTON_FONT)
        self.match_button.clicked.connect(self._match_button_clicked)
        self.buttons_widget_layout.addWidget(self.match_button)

        self.no_match_button = QPushButton("There Is No Match")
        self.no_match_button.setFont(BUTTON_FONT)
        self.no_match_button.clicked.connect(self._no_match_button_clicked)
        self.buttons_widget_layout.addWidget(self.no_match_button)

    def _left_button_clicked(self):
        idx = self.nuggets_in_order.index(self.current_nugget)
        if idx > 0:
            self.current_nugget = self.nuggets_in_order[idx - 1]
            self._highlight_current_nugget()
            self.suggestion_list.update_item_list(self.nuggets_sorted_by_distance, self.current_nugget)

    def _right_button_clicked(self):
        idx = self.nuggets_in_order.index(self.current_nugget)
        if idx < len(self.nuggets_in_order) - 1:
            self.current_nugget = self.nuggets_in_order[idx + 1]
            self._highlight_current_nugget()
            self.suggestion_list.update_item_list(self.nuggets_sorted_by_distance, self.current_nugget)

    def _match_button_clicked(self):
        self.interactive_matching_widget.main_window.give_feedback_task({
            "message": "is-match",
            "nugget": self.current_nugget
        })

    def _no_match_button_clicked(self):
        self.interactive_matching_widget.main_window.give_feedback_task({
            "message": "no-match-in-document",
            "nugget": self.current_nugget
        })

    def _highlight_current_nugget(self):
        mapped_start_char = self.idx_mapper[self.current_nugget.start_char]
        mapped_end_char = self.idx_mapper[self.current_nugget.end_char]
        formatted_text = (
            f"{self.base_formatted_text[:mapped_start_char]}"
            f"<span style='background-color: {YELLOW}'><b>"
            f"{self.base_formatted_text[mapped_start_char:mapped_end_char]}</span></b>"
            f"{self.base_formatted_text[mapped_end_char:]}"
        )
        self.text_edit.setText("")
        self.text_edit.textCursor().insertHtml(formatted_text)

    def update_document(self, nugget):
        self.document = nugget.document
        self.current_nugget = nugget
        self.nuggets_sorted_by_distance = list(sorted(self.document.nuggets, key=lambda x: x[CachedDistanceSignal]))
        self.nuggets_in_order = list(sorted(self.document.nuggets, key=lambda x: x.start_char))

        if self.nuggets_in_order != []:
            self.idx_mapper = {}
            char_list = []
            end_chars = []
            next_start_char = self.nuggets_in_order[0].start_char
            next_nugget_idx = 0
            for idx, char in enumerate(list(self.document.text)):
                if idx == next_start_char:
                    char_list += list("</span></b>")
                    char_list += list(f"<span style='background-color: {LIGHT_YELLOW}'><b>")
                    end_chars.append(self.nuggets_in_order[next_nugget_idx].end_char)
                    next_nugget_idx += 1
                    if next_nugget_idx < len(self.nuggets_in_order):
                        next_start_char = self.nuggets_in_order[next_nugget_idx].start_char
                    else:
                        next_start_char = -1
                while idx in end_chars:
                    end_chars.remove(idx)
                if end_chars == []:
                    char_list += list("</span></b>")
                self.idx_mapper[idx] = len(char_list)
                char_list.append(char)
            self.base_formatted_text = "".join(char_list)
        else:
            self.idx_mapper = {}
            for idx in range(len(self.document.text)):
                self.idx_mapper[idx] = idx
            self.base_formatted_text = ""

        self._highlight_current_nugget()

        scroll_cursor = QTextCursor(self.text_edit.document())
        scroll_cursor.setPosition(nugget.start_char)
        self.text_edit.setTextCursor(scroll_cursor)
        self.text_edit.ensureCursorVisible()

        self.suggestion_list.update_item_list(self.nuggets_sorted_by_distance, self.current_nugget)

    def enable_input(self):
        self.left_button.setEnabled(True)
        self.right_button.setEnabled(True)
        self.match_button.setEnabled(True)
        self.no_match_button.setEnabled(True)
        self.suggestion_list.enable_input()

    def disable_input(self):
        self.left_button.setDisabled(True)
        self.right_button.setDisabled(True)
        self.match_button.setDisabled(True)
        self.no_match_button.setDisabled(True)
        self.suggestion_list.disable_input()


class SuggestionListItemWidget(CustomScrollableListItem):

    def __init__(self, suggestion_list_widget):
        super(SuggestionListItemWidget, self).__init__(suggestion_list_widget)
        self.suggestion_list_widget = suggestion_list_widget
        self.nugget = None

        self.setFixedHeight(30)
        self.setStyleSheet(f"background-color: {WHITE}")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 20, 0)
        self.layout.setSpacing(10)

        self.text_label = QLabel()
        self.text_label.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.text_label)

        self.split_label = QLabel("|")
        self.split_label.setFont(CODE_FONT_BOLD)
        self.layout.addWidget(self.split_label)

        self.choose_button = QPushButton()
        self.choose_button.setIcon(QIcon("aset_ui/resources/correct.svg"))
        self.choose_button.setFlat(True)
        self.choose_button.clicked.connect(self._choose_button_clicked)
        self.layout.addWidget(self.choose_button)

    def update_item(self, item, params=None):
        self.nugget = item
        self.text_label.setText(self.nugget.text)
        if self.nugget == params:
            self.setStyleSheet(f"background-color: {YELLOW}")
        else:
            self.setStyleSheet(f"background-color: {LIGHT_YELLOW}")

    def _choose_button_clicked(self):
        self.suggestion_list_widget.interactive_matching_widget.main_window.give_feedback_task({
            "message": "is-match",
            "nugget": self.nugget
        })

    def enable_input(self):
        self.choose_button.setEnabled(True)

    def disable_input(self):
        self.choose_button.setDisabled(True)
