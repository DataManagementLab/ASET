"""
Matching step.

The matching stage is responsible for the matching step. The matching process comprises two objectives. First, the
attributes of the user's schema must be embedded. Afterward, the matching stage matches the extracted information
nuggets against the schema. Depending on the matching strategy, this step may also involve gathering user feedback and
updating the attribute embeddings accordingly before the actual matching between the extractions and attributes starts.
The matching stage's relevant data model components are the attributes, rows, documents, extractions, and extraction and
attribute embeddings.
"""
import logging
from time import time

from aset.core.json_serialization import Component
from aset.embedding.aggregation import AttributeEmbeddingMethod
from aset.extraction.common import Document
from aset.matching.common import Attribute, Row
from aset.matching.strategies import BaseStrategy

logger = logging.getLogger(__name__)


class MatchingStage(Component):
    """Matching stage that handles the matching step."""

    def __init__(self,
                 documents: [Document],
                 attributes: [Attribute],
                 strategy: BaseStrategy or None,
                 embedding_method: AttributeEmbeddingMethod or None):
        """Initialize the matching stage with the given documents, attributes, strategy and embedding method."""
        logger.info("Prepare the matching stage with {} documents, {} and '{}' and '{}'.".format(
            len(documents),
            [attributes.label for attributes in attributes],
            strategy.strategy_str if strategy is not None else '-',
            embedding_method.embedding_method_str if embedding_method is not None else '-'
        ))

        tik = time()

        self.attributes: [Attribute] = attributes
        self.documents: [Document] = documents
        self.rows: [Row] = [Row(self.attributes) for _ in documents]
        self.strategy: BaseStrategy = strategy
        self.embedding_method: AttributeEmbeddingMethod = embedding_method

        tak = time()

        logger.info(f"Prepared the matching stage in {tak - tik} seconds.")

    def __str__(self):
        return "Documents: {}\nRows: {}\nAttributes: {}\nStrategy: {}\nEmbedding Method: {}".format(
            len(self.documents),
            len(self.rows),
            [attribute.label for attribute in self.attributes],
            self.strategy.strategy_str if self.strategy is not None else '-',
            self.embedding_method if self.embedding_method is not None else '-')

    def __eq__(self, other):
        return other is not None \
               and len(self.documents) == len(other.documents) \
               and all(a == b for a, b in zip(self.documents, other.documents)) \
               and self.strategy == other.strategy \
               and len(self.attributes) == len(other.attributes) \
               and all(a == b for a, b in zip(self.attributes, other.attributes)) \
               and len(self.rows) == len(other.rows) \
               and all(a == b for a, b in zip(self.rows, other.rows)) \
               and self.embedding_method == other.embedding_method

    def compute_attribute_embeddings(self):
        """Compute embeddings for the attributes."""
        logger.info("Compute embeddings for {} attributes with '{}'".format(
            len(self.attributes),
            self.embedding_method.embedding_method_str if self.embedding_method is not None else '-'
        ))

        tik = time()

        # compute the embeddings
        if self.embedding_method is None:
            logger.error("No embedding method has been defined!")
        else:
            self.embedding_method(self.attributes)

        tak = time()

        logger.info(f"Computed embeddings in {tak - tik} seconds.")

    def incorporate_example_mentions(self, mentions_by_attribute: [[str]]):
        """Update the attribute embeddings with the provided mentions."""
        logger.info("Update the attribute embeddings with the user-provided examples.")

        tik = time()

        if self.embedding_method is None:
            logger.error("No embedding method has been defined!")
        else:
            for attribute, mentions in zip(self.attributes, mentions_by_attribute):
                embeddings = self.embedding_method.compute_mention_embeddings(mentions)
                attribute.embedding.update(embeddings)

        tak = time()

        logger.info(f"Updated attributes in {tak - tik} seconds.")

    def match_extractions_to_attributes(self):
        """Match extractions to attributes with matching strategy."""
        logger.info("Match extractions from {} documents to {} with '{}' and '{}'".format(
            len(self.documents),
            [attribute.label for attribute in self.attributes],
            self.strategy.strategy_str if self.strategy is not None else '-',
            self.embedding_method.embedding_method_str if self.embedding_method is not None else '-'
        ))

        tik = time()

        # match the extractions to the rows
        if len(self.attributes) == 0:
            logger.error("There are no attributes the extractions could be matched to!")
        elif self.strategy is None:
            logger.error("No strategy has been defined!")
        else:
            self.rows = self.strategy(self.documents, self.attributes)

        tak = time()

        logger.info(f"Matched extractions in {tak - tik} seconds.")

    @property
    def table_str(self):
        """Schema and rows as a table string representation."""
        lines = ["".join(["{:50.50}".format(attribute.label) for attribute in self.attributes])]
        lines += [row.row_str for row in self.rows]
        return "\n".join(lines)

    @property
    def json_dict(self):
        return {
            "attributes": [attribute.json_dict for attribute in self.attributes],
            "documents": [document.json_dict for document in self.documents],
            "rows": [row.json_dict for row in self.rows]
        }

    @classmethod
    def from_json_dict(cls, dictionary: dict):
        documents = [Document.from_json_dict(d) for d in dictionary["documents"]]
        attributes = [Attribute.from_json_dict(d) for d in dictionary["attributes"]]
        matching_stage = cls(documents, attributes, None, None)

        # load the given rows
        matching_stage.rows = [Row.from_json_dict(d) for d in dictionary["rows"]]

        # does not restore the matching strategy or embedding method
        return matching_stage
