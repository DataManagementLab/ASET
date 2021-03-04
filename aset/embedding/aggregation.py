"""
Aggregation of the embeddings of the individual signals.

They define the aggregation of the individual signals with
regards to distance calculation and updating based on in-column and out-of-column embeddings. Furthermore, they handle
the JSON serialization.
"""
import logging

import numpy as np
from scipy.spatial.distance import cosine

from aset.core.json_serialization import Component
from aset.core.resources import get_stanza_tokenize_pipeline, get_fasttext_embedding, get_sentence_bert, \
    get_bert_tokenizer_device
from aset.embedding.signals import labels_as_fasttext_natural_language_embeddings, sbert_natural_language_embeddings, \
    positions_as_relative_positions, context_with_bert_natural_language_embeddings, \
    labels_as_sbert_natural_language_embeddings

logger = logging.getLogger(__name__)


class Embedding(Component):
    """Embeddings for attributes and extractions."""

    def __init__(
            self,
            label_embedding: np.array or None,
            mention_embedding: np.array or None,
            context_embedding: np.array or None,
            position_embedding: int or None
    ):
        """Create an embedding with the given signal embeddings."""
        self.label_embedding: np.array or None = label_embedding
        if self.label_embedding is not None:
            self.num_label_embedding: int = 1
        else:
            self.num_label_embedding: int = 0

        self.mention_embedding: np.array or None = mention_embedding
        if self.mention_embedding is not None:
            self.num_mention_embedding: int = 1
        else:
            self.num_mention_embedding: int = 0

        self.context_embedding: np.array or None = context_embedding
        if self.context_embedding is not None:
            self.num_context_embedding: int = 1
        else:
            self.num_context_embedding: int = 0

        self.position_embedding: int or None = position_embedding
        if self.position_embedding is not None:
            self.num_position_embedding: int = 1
        else:
            self.num_position_embedding: int = 0

    # noinspection PyTypeChecker
    def __eq__(self, other):
        if other is None:
            return False

        if self.num_label_embedding != other.num_label_embedding:
            return False
        if self.label_embedding is not None and other.label_embedding is not None:
            if not all(self.label_embedding == other.label_embedding):
                return False
        elif not (self.label_embedding is None and other.label_embedding is None):
            return False

        if self.num_mention_embedding != other.num_mention_embedding:
            return False
        if self.mention_embedding is not None and other.mention_embedding is not None:
            if not all(self.mention_embedding == other.mention_embedding):
                return False
        elif not (self.mention_embedding is None and other.mention_embedding is None):
            return False

        if self.num_context_embedding != other.num_context_embedding:
            return False
        if self.context_embedding is not None and other.context_embedding is not None:
            if not all(self.context_embedding == other.context_embedding):
                return False
        elif not (self.context_embedding is None and other.context_embedding is None):
            return False

        if self.num_position_embedding != other.num_position_embedding:
            return False
        if not self.position_embedding == other.position_embedding:
            return False

        return True

    def distance(self, embedding):
        """Calculate the distance to the given embedding."""
        distance = 0
        num_pairs = 0

        # cosine distance between the label embeddings
        if self.label_embedding is not None and embedding.label_embedding is not None:
            distance += min(abs(cosine(self.label_embedding, embedding.label_embedding)), 1)
            num_pairs += 1

        # cosine distance between the mention embeddings
        if self.mention_embedding is not None and embedding.mention_embedding is not None:
            distance += min(abs(cosine(self.mention_embedding, embedding.mention_embedding)), 1)
            num_pairs += 1

        # cosine distance between the context embeddings
        if self.context_embedding is not None and embedding.context_embedding is not None:
            distance += min(abs(cosine(self.context_embedding, embedding.context_embedding)), 1)
            num_pairs += 1

        # absolute distance between the position embeddings
        if self.position_embedding is not None and embedding.position_embedding is not None:
            distance += min(abs(self.position_embedding - embedding.position_embedding), 1)
            num_pairs += 1

        # aggregate the distances between the individual signals
        assert num_pairs != 0, "Unable to calculate distance since the embeddings do not share signals!"
        return distance / num_pairs

    def update(self, in_column_embeddings):
        """Update the embedding as the average of the in-column embeddings."""

        def average(embeddings, previous, num_previous):
            embeddings = [e for e in embeddings if e is not None]
            if len(embeddings) == 0:
                return previous, num_previous
            else:
                if previous is None:
                    return sum(embeddings) / len(embeddings), len(embeddings)
                else:
                    total = sum(embeddings) + num_previous * previous
                    divisor = len(embeddings) + num_previous
                    return total / divisor, divisor

        label_embeddings = [e.label_embedding for e in in_column_embeddings]
        self.label_embedding, self.num_label_embedding = \
            average(label_embeddings, self.label_embedding, self.num_label_embedding)

        mention_embeddings = [e.mention_embedding for e in in_column_embeddings]
        self.mention_embedding, self.num_mention_embedding = \
            average(mention_embeddings, self.mention_embedding, self.num_mention_embedding)

        context_embeddings = [e.context_embedding for e in in_column_embeddings]
        self.context_embedding, self.num_context_embedding = \
            average(context_embeddings, self.context_embedding, self.num_context_embedding)

        position_embeddings = [e.position_embedding for e in in_column_embeddings]
        self.position_embedding, self.num_position_embedding = \
            average(position_embeddings, self.position_embedding, self.num_position_embedding)

    @property
    def json_dict(self):
        # numpy arrays are stored as lists
        serialized_label_embedding = self.label_embedding.tolist() if self.label_embedding is not None else None
        serialized_mention_embedding = self.mention_embedding.tolist() if self.mention_embedding is not None else None
        serialized_context_embedding = self.context_embedding.tolist() if self.context_embedding is not None else None

        return {
            "label_embedding": serialized_label_embedding,
            "num_label_embedding": self.num_label_embedding,
            "mention_embedding": serialized_mention_embedding,
            "num_mention_embedding": self.num_mention_embedding,
            "context_embedding": serialized_context_embedding,
            "num_context_embedding": self.num_context_embedding,
            "position_embedding": self.position_embedding,
            "num_position_embedding": self.num_position_embedding
        }

    @classmethod
    def from_json_dict(cls, values: dict):
        # numpy arrays are stored as lists
        label_embedding = np.array(lst) if (lst := values["label_embedding"]) is not None else None
        mention_embedding = np.array(lst) if (lst := values["mention_embedding"]) is not None else None
        context_embedding = np.array(lst) if (lst := values["context_embedding"]) is not None else None

        embedding = cls(
            label_embedding,
            mention_embedding,
            context_embedding,
            values['position_embedding']
        )
        embedding.num_mention_embedding = values["num_mention_embedding"]
        embedding.num_label_embedding = values["num_label_embedding"]
        embedding.num_context_embedding = values["num_context_embedding"]
        embedding.num_position_embedding = values["num_position_embedding"]

        return embedding


class AttributeEmbeddingMethod:
    """Embedding method for embedding attributes."""

    embedding_method_str = "AttributeEmbeddingMethod"

    def __init__(self):
        # preload the required resources
        get_stanza_tokenize_pipeline()
        get_fasttext_embedding()
        get_sentence_bert()
        get_bert_tokenizer_device()

    @staticmethod
    def __call__(attributes):
        label_embeddings = labels_as_fasttext_natural_language_embeddings([attr.label for attr in attributes])
        mention_embeddings = [None for _ in attributes]
        context_embeddings = [None for _ in attributes]
        position_embeddings = [None for _ in attributes]

        for i, attribute in enumerate(attributes):
            attribute.embedding = Embedding(
                label_embeddings[i],
                mention_embeddings[i],
                context_embeddings[i],
                position_embeddings[i]
            )

    @staticmethod
    def compute_mention_embeddings(mentions: [str]):
        label_embeddings = [None for _ in mentions]
        mention_embeddings = sbert_natural_language_embeddings(mentions)
        context_embeddings = [None for _ in mentions]
        position_embeddings = [None for _ in mentions]

        embeddings = []
        for i in range(len(label_embeddings)):
            embeddings.append(Embedding(
                label_embeddings[i],
                mention_embeddings[i],
                context_embeddings[i],
                position_embeddings[i]
            ))

        return embeddings


class ExtractionEmbeddingMethod:
    """Embedding method for embedding extractions."""

    embedding_method_str = "ExtractionEmbeddingMethod"

    def __init__(self):
        # preload the required resources
        get_stanza_tokenize_pipeline()
        get_fasttext_embedding()
        get_sentence_bert()
        get_bert_tokenizer_device()

    def __eq__(self, other):
        return self.embedding_method_str == other.embedding_method_str

    @staticmethod
    def __call__(documents):
        for i, document in enumerate(documents):
            if i % (len(documents) // 5) == 0:
                logger.info("Computing extraction embeddings {} percent done.".format(round(i / len(documents) * 100)))

            extractions = document.extractions

            label_embeddings = labels_as_fasttext_natural_language_embeddings([ext.label for ext in extractions])
            mention_embeddings = sbert_natural_language_embeddings([ext.mention for ext in extractions])
            context_embeddings = context_with_bert_natural_language_embeddings(
                [ext.context for ext in extractions],
                [ext.mention for ext in extractions]
            )
            position_embeddings = positions_as_relative_positions([ext.position for ext in extractions])

            for extraction_i, extraction in enumerate(extractions):
                extraction.embedding = Embedding(
                    label_embeddings[extraction_i],
                    mention_embeddings[extraction_i],
                    context_embeddings[extraction_i],
                    position_embeddings[extraction_i]
                )
