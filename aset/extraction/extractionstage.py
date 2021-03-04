"""
Extraction Step.

The extraction stage handles the extraction step. In essence, it performs three tasks. First, it derives the information
nuggets from the documents. The second objective is to determine the actual data values of the derived extractions.
Finally, it computes the embeddings for the extractions, which the matching process later uses to match between
extractions and attributes. The relevant data model components in the extraction stage are the documents, extractions,
and extraction embeddings.
"""

import logging
from time import time

from aset.core.json_serialization import Component
from aset.embedding.aggregation import ExtractionEmbeddingMethod
from aset.extraction.common import Document
from aset.extraction.extractors import BaseExtractor
from aset.extraction.processors import BaseProcessor

logger = logging.getLogger(__name__)


class ExtractionStage(Component):
    """Extraction stage that handles the extraction step."""

    def __init__(self, documents: [Document],
                 extractors: [BaseExtractor],
                 processors: [BaseProcessor],
                 embedding_method: ExtractionEmbeddingMethod or None):
        """Initialize the extraction stage with the given documents, extractors, processors and embedding method."""
        logger.info("Prepare the extraction stage with {} documents, {}, {}, and '{}'.".format(
            len(documents),
            [extractor.extractor_str for extractor in extractors],
            [processor.processor_str for processor in processors],
            embedding_method.embedding_method_str if embedding_method is not None else "-"
        ))

        tik = time()

        self.documents: [Document] = documents
        self.extractors: [BaseExtractor] = extractors
        self.processors: [BaseProcessor] = processors
        self.embedding_method: ExtractionEmbeddingMethod = embedding_method

        tak = time()

        logger.info(f"Prepared the extraction stage in {tak - tik} seconds.")

    def __str__(self):
        total_extractions = sum(len(document.extractions) for document in self.documents)

        return "Extractors: {}\nProcessors: {}\nEmbedding Method: {}\nDocuments: {}\nExtractions: {}".format(
            [extractor.extractor_str for extractor in self.extractors],
            [processor.processor_str for processor in self.processors],
            self.embedding_method.embedding_method_str if self.embedding_method is not None else "-",
            len(self.documents),
            total_extractions)

    def __eq__(self, other):
        return other is not None \
               and len(self.extractors) == len(other.extractors) \
               and all(a == b for a, b in zip(self.extractors, other.extractors)) \
               and len(self.processors) == len(other.processors) \
               and all(a == b for a, b in zip(self.processors, other.processors)) \
               and self.embedding_method == other.embedding_method \
               and len(self.documents) == len(other.documents) \
               and all(a == b for a, b in zip(self.documents, other.documents))

    def derive_extractions(self):
        """Derive extractions from the documents."""
        logger.info("Derive extractions from {} documents with {}.".format(
            len(self.documents),
            [extractor.extractor_str for extractor in self.extractors]
        ))

        tik = time()
        for extractor in self.extractors:
            extractor(self.documents)

        tak = time()

        logger.info(f"Derived extractions in {tak - tik} seconds.")

    def determine_values(self):
        """Determine the values of the extractions."""
        logger.info("Determine the values of {} extractions with {}.".format(
            sum(map(lambda doc: len(doc.extractions), self.documents)),
            [processor.processor_str for processor in self.processors]
        ))

        tik = time()

        # gather the extractions from all documents
        extractions = []
        for document in self.documents:
            extractions += document.extractions

        # determine the actual values
        for processor in self.processors:
            processor(extractions)

        tak = time()

        logger.info(f"Determined values in {tak - tik} seconds.")

    def compute_extraction_embeddings(self):
        """Compute the embeddings of the extractions."""
        logger.info("Compute the embeddings of {} extractions with '{}'".format(
            sum(map(lambda doc: len(doc.extractions), self.documents)),
            self.embedding_method.embedding_method_str if self.embedding_method is not None else '-'
        ))

        tik = time()

        if self.embedding_method is None:
            logger.error("No embedding method has been defined!")
        else:
            self.embedding_method(self.documents)

        tak = time()

        logger.info(f"Computed embeddings in {tak - tik} seconds.")

    @property
    def json_dict(self):
        return {
            "documents": [document.json_dict for document in self.documents]
        }

    @classmethod
    def from_json_dict(cls, values: dict):
        documents = [Document.from_json_dict(d) for d in values["documents"]]

        # does not restore the extractors, processors or embedding method
        return cls(documents, [], [], None)
