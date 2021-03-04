"""Data model of the extraction stage."""
import json
import logging

from aset.core.json_serialization import Component
from aset.embedding.aggregation import Embedding

logger = logging.getLogger(__name__)


class Document(Component):
    """Document provided to the extraction stage."""

    def __init__(self, text: str):
        """
        Create a document.

        :param text: natural language text of the document
        """
        self.text: str = text
        self.extractions: [Extraction] = []

    def __str__(self):
        return json.dumps(self.json_dict, indent=4)

    def __eq__(self, other):
        return other is not None \
               and self.text == other.text \
               and len(self.extractions) == len(other.extractions) \
               and all(a == b for a, b in zip(self.extractions, other.extractions))

    @property
    def json_dict(self):
        return {
            "text": self.text,
            "extractions": [ext.json_dict for ext in self.extractions]
        }

    @classmethod
    def from_json_dict(cls, values: dict):
        document = cls(values["text"])

        # load the given extractions and do not recalculate them
        document.extractions = [Extraction.from_json_dict(e) for e in values["extractions"]]

        return document


class Extraction(Component):
    """Information nugget derived from a document."""

    # extraction type identifiers
    string_extraction_type_str = "STRING"
    datetime_extraction_type_str = "DATETIME"
    number_extraction_type_str = "NUMBER"

    def __init__(self,
                 extraction_type_str: str,
                 extractor_str: str,
                 label: str,
                 mention: str,
                 context: str,
                 position: int):
        """
        Create an extraction.

        :param extraction_type_str: identifies the type of extraction
        :param extractor_str: identifiers the extractor this extraction comes from
        :param label: meta-information
        :param mention: natural language mention of the value in the text
        :param context: sentence the mention appears in
        :param position: position of the first token in the document
        """
        self.extraction_type_str: str = extraction_type_str
        self.extractor_str: str = extractor_str
        self.label: str = label
        self.mention: str = mention
        self.context: str = context
        self.position: int = position

        self.value: str or None = None
        self.embedding: Embedding or None = None

    def __str__(self):
        return json.dumps(self.json_dict, indent=4)

    def __eq__(self, other):
        return other is not None \
               and self.extraction_type_str == other.extraction_type_str \
               and self.extractor_str == other.extractor_str \
               and self.label == other.label \
               and self.value == other.value \
               and self.mention == other.mention \
               and self.context == other.context \
               and self.position == other.position \
               and self.embedding == other.embedding

    @property
    def json_dict(self):
        return {
            "extraction_type_str": self.extraction_type_str,
            "extractor_str": self.extractor_str,
            "label": self.label,
            "value": self.value,
            "mention": self.mention,
            "position": self.position,
            "context": self.context,
            "embedding": self.embedding.json_dict if self.embedding is not None else None
        }

    @classmethod
    def from_json_dict(cls, values: dict):
        extraction = cls(values["extraction_type_str"],
                         values["extractor_str"],
                         values["label"],
                         values["mention"],
                         values["context"],
                         values["position"])

        # load the given value and do not recalculate it
        if values["value"] is not None:
            extraction.value = values["value"]

        # load the given embedding and do not recalculate it
        if values["embedding"] is not None:
            extraction.embedding = Embedding.from_json_dict(values["embedding"])

        return extraction
