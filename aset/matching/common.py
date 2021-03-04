"""Data model for the matching stage."""
import json
import logging

from aset.core.json_serialization import Component
from aset.embedding.aggregation import Embedding
from aset.extraction.common import Extraction

logger = logging.getLogger(__name__)


class Attribute(Component):
    """Attribute in the output table."""

    def __init__(self, label: str):
        """Create an attribute with the given label."""
        self.label = label

        self.embedding = None

    def __eq__(self, other):
        return self.label == other.label and self.embedding == other.embedding

    def __str__(self):
        return json.dumps(self.json_dict, indent=4)

    @property
    def json_dict(self):
        return {
            "label": self.label,
            "embedding": self.embedding.json_dict if self.embedding is not None else None
        }

    @classmethod
    def from_json_dict(cls, values: dict):
        attribute = cls(values["label"])

        # load the given attribute embedding
        if values["embedding"] is not None:
            attribute.embedding = Embedding.from_json_dict(values["embedding"])

        return attribute


class Row(Component):
    """Row in the output table."""

    def __init__(self, attributes: [Attribute]):
        """Create a row for the given attributes."""
        self.extractions: {str: Extraction} = {attribute.label: None for attribute in attributes}

    def __eq__(self, other):
        return self.extractions == other.extractions

    def __str__(self):
        return json.dumps(self.json_dict, indent=4)

    @property
    def row_str(self):
        """String representation of the row that can be used in a table."""
        strings = []
        for extraction in self.extractions.values():
            if extraction is None:  # no match has been found
                strings.append("[no-match]")
            elif extraction.value is None:  # the extraction has no value
                strings.append("[no-value]")
            else:
                strings.append(str(extraction.value).replace("\n", " "))

        return "".join("{:50.50}".format(string) for string in strings)

    @property
    def json_dict(self):
        return {
            "extractions": {label: e.json_dict if e is not None else None for label, e in self.extractions.items()}
        }

    @classmethod
    def from_json_dict(cls, dictionary: dict):
        row = cls([Attribute(label) for label in dictionary['extractions'].keys()])
        row.extractions = {}
        for label, extraction in dictionary["extractions"].items():
            if extraction is not None:
                row.extractions[label] = Extraction.from_json_dict(extraction)
            else:
                row.extractions[label] = None
        return row
