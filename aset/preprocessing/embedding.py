import abc
import logging
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from aset import resources
from aset.configuration import BasePipelineElement, register_configurable_element
from aset.data.data import ASETAttribute, ASETDocumentBase, ASETNugget
from aset.data.signals import ContextSentenceEmbeddingSignal, LabelEmbeddingSignal, RelativePositionSignal, \
    TextEmbeddingSignal, UserProvidedExamplesSignal, NaturalLanguageLabelSignal, CachedContextSentenceSignal
from aset.interaction import BaseInteractionCallback
from aset.statistics import Statistics
from aset.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class BaseEmbedder(BasePipelineElement, abc.ABC):
    """
    Base class for all embedders.

    Embedders work with nuggets and attributes and transform their signals and other information into embedding signals.
    """
    identifier: str = "BaseEmbedder"

    def _use_status_callback_for_embedder(
            self,
            status_callback: BaseStatusCallback,
            element: str,
            ix: int,
            total: int
    ) -> None:
        """
        Helper method that calls the status callback at regular intervals.

        :param status_callback: status callback to call
        :param element: 'nuggets' or 'attributes'
        :param ix: index of the current element
        :param total: total number of elements
        """
        if total == 0:
            status_callback(f"Embedding {element} with {self.identifier}...", -1)
        elif ix == 0:
            status_callback(f"Embedding {element} with {self.identifier}...", 0)
        else:
            interval: int = total // 10
            if interval != 0 and ix % interval == 0:
                status_callback(f"Embedding {element} with {self.identifier}...", ix / total)

    def _call(
            self,
            document_base: ASETDocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        # compute embeddings for the nuggets
        nuggets: List[ASETNugget] = document_base.nuggets
        logger.info(f"Embed {len(nuggets)} nuggets with {self.identifier}.")
        tick: float = time.time()
        status_callback(f"Embedding nuggets with {self.identifier}...", -1)
        statistics["nuggets"]["num_nuggets"] = len(nuggets)
        self._embed_nuggets(nuggets, interaction_callback, status_callback, statistics["nuggets"])
        status_callback(f"Embedding nuggets with {self.identifier}...", 1)
        tack: float = time.time()
        logger.info(f"Embedded {len(nuggets)} nuggets with {self.identifier} in {tack - tick} seconds.")
        statistics["nuggets"]["runtime"] = tack - tick

        # compute embeddings for the attributes
        attributes: List[ASETAttribute] = document_base.attributes
        logger.info(f"Embed {len(attributes)} attributes with {self.identifier}.")
        tick: float = time.time()
        status_callback(f"Embedding attributes with {self.identifier}...", -1)
        statistics["attributes"]["num_attributes"] = len(attributes)
        self._embed_attributes(attributes, interaction_callback, status_callback, statistics["attributes"])
        status_callback(f"Embedding attributes with {self.identifier}...", 1)
        tack: float = time.time()
        logger.info(f"Embedded {len(attributes)} attributes with {self.identifier} in {tack - tick} seconds.")
        statistics["attributes"]["runtime"] = tack - tick

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        """
        Compute embeddings for the given list of nuggets.

        :param nuggets: list of nuggets to work on
        :param interaction_callback: callback to allow for user interaction
        :param status_callback: callback to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing

    def _embed_attributes(
            self,
            attributes: List[ASETAttribute],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        """
        Compute embeddings for the given list of ASETNuggets.

        :param attributes: list of ASETAttributes to work on
        :param interaction_callback: callback to allow for user interaction
        :param status_callback: callback to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing


########################################################################################################################
# actual embedders
########################################################################################################################


class BaseSBERTEmbedder(BaseEmbedder, abc.ABC):
    """Base class for all embedders based on SBERT."""
    identifier: str = "BaseSBERTEmbedder"

    def __init__(self, sbert_resource_identifier: str) -> None:
        """
        Initialize the embedder.

        :param sbert_resource_identifier: identifier of the SBERT model resource
        """
        super(BaseSBERTEmbedder, self).__init__()
        self._sbert_resource_identifier: str = sbert_resource_identifier

        # preload required resources
        resources.MANAGER.load(self._sbert_resource_identifier)
        logger.debug(f"Initialized '{self.identifier}'.")

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "sbert_resource_identifier": self._sbert_resource_identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseSBERTEmbedder":
        return cls(config["sbert_resource_identifier"])


@register_configurable_element
class SBERTLabelEmbedder(BaseSBERTEmbedder):
    """Label embedder based on SBERT."""
    identifier: str = "SBERTLabelEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [NaturalLanguageLabelSignal.identifier],
        "attributes": [NaturalLanguageLabelSignal.identifier],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelEmbeddingSignal.identifier],
        "attributes": [LabelEmbeddingSignal.identifier],
        "documents": []
    }

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [nugget[NaturalLanguageLabelSignal] for nugget in nuggets]
        embeddings: List[np.ndarray] = resources.MANAGER[self._sbert_resource_identifier].encode(
            texts, show_progress_bar=False
        )

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[LabelEmbeddingSignal] = LabelEmbeddingSignal(embedding)

    def _embed_attributes(
            self,
            attributes: List[ASETAttribute],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [attribute[NaturalLanguageLabelSignal] for attribute in attributes]
        embeddings: List[np.ndarray] = resources.MANAGER[self._sbert_resource_identifier].encode(
            texts, show_progress_bar=False
        )

        for attribute, embedding in zip(attributes, embeddings):
            attribute[LabelEmbeddingSignal] = LabelEmbeddingSignal(embedding)


@register_configurable_element
class SBERTTextEmbedder(BaseSBERTEmbedder):
    """Text embedder based on SBERT."""
    identifier: str = "SBERTTextEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [TextEmbeddingSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [nugget.text for nugget in nuggets]
        embeddings: List[np.ndarray] = resources.MANAGER[self._sbert_resource_identifier].encode(
            texts, show_progress_bar=False
        )

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[TextEmbeddingSignal] = TextEmbeddingSignal(embedding)


@register_configurable_element
class SBERTExamplesEmbedder(BaseSBERTEmbedder):
    """SBERT-based embedder to embed user-provided example texts."""
    identifier: str = "SBERTExamplesEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [UserProvidedExamplesSignal.identifier],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [TextEmbeddingSignal.identifier],
        "documents": []
    }

    def _embed_attributes(
            self,
            attributes: List[ASETAttribute],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        for ix, attribute in enumerate(attributes):
            self._use_status_callback_for_embedder(status_callback, "attributes", ix, len(attributes))
            texts: List[str] = attribute[UserProvidedExamplesSignal]
            if texts != []:
                embeddings: List[np.ndarray] = resources.MANAGER[self._sbert_resource_identifier].encode(
                    texts, show_progress_bar=False
                )
                embedding: np.ndarray = np.mean(embeddings, axis=0)
                attribute[TextEmbeddingSignal] = TextEmbeddingSignal(embedding)
                statistics["num_has_examples"] += 1
            else:
                statistics["num_no_examples"] += 1


@register_configurable_element
class SBERTContextSentenceEmbedder(BaseSBERTEmbedder):
    """Context sentence embedder based on SBERT."""
    identifier: str = "SBERTContextSentenceEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [CachedContextSentenceSignal.identifier],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [ContextSentenceEmbeddingSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [nugget[CachedContextSentenceSignal]["text"] for nugget in nuggets]

        # compute embeddings
        embeddings: List[np.ndarray] = resources.MANAGER[self._sbert_resource_identifier].encode(
            texts, show_progress_bar=False
        )

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[ContextSentenceEmbeddingSignal] = ContextSentenceEmbeddingSignal(embedding)


@register_configurable_element
class BERTContextSentenceEmbedder(BaseEmbedder):
    """
    Context sentence embedder based on BERT.

    Computes the context embedding of an ASETNugget as the mean of the final hidden states of the tokens that make up
    the nugget in its context sentence.
    """
    identifier: str = "BERTContextSentenceEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [CachedContextSentenceSignal.identifier],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [ContextSentenceEmbeddingSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def __init__(self, bert_resource_identifier: str) -> None:
        """
        Initialize the BERTContextSentenceEmbedder.

        :param bert_resource_identifier: identifier of the BERT model resource
        """
        super(BERTContextSentenceEmbedder, self).__init__()
        self._bert_resource_identifier: str = bert_resource_identifier

        # preload required resources
        resources.MANAGER.load(self._bert_resource_identifier)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:

        if resources.MANAGER[self._bert_resource_identifier]["device"] is not None:
            resources.MANAGER[self._bert_resource_identifier]["model"].to(
                resources.MANAGER[self._bert_resource_identifier]["device"]
            )

        for nugget_ix, nugget in enumerate(nuggets):
            self._use_status_callback_for_embedder(status_callback, "nuggets", nugget_ix, len(nuggets))

            context_sentence: str = nugget[CachedContextSentenceSignal]["text"]
            start_in_context: int = nugget[CachedContextSentenceSignal]["start_char"]
            end_in_context: int = nugget[CachedContextSentenceSignal]["end_char"]

            # compute the sentence's token embeddings
            encoding = resources.MANAGER[self._bert_resource_identifier]["tokenizer"](
                context_sentence, return_tensors="pt", padding=True
            )

            device = resources.MANAGER[self._bert_resource_identifier]["device"]
            if device is not None:
                input_ids = encoding.input_ids.to(device)
                token_type_ids = encoding.token_type_ids.to(device)
                attention_mask = encoding.attention_mask.to(device)
            else:
                input_ids = encoding.input_ids
                token_type_ids = encoding.token_type_ids
                attention_mask = encoding.attention_mask

            outputs = resources.MANAGER[self._bert_resource_identifier]["model"](
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            torch_output = outputs[0].detach()
            if device is not None:
                torch_output = torch_output.cpu()
            output: np.ndarray = torch_output[0].numpy()

            # determine which tokens make up the nugget
            token_indices_set: Set[int] = set()
            for char_ix in range(start_in_context, end_in_context):
                token_ix: Optional[int] = encoding.char_to_token(char_ix)
                if token_ix is not None:
                    token_indices_set.add(token_ix)
            token_indices: List[int] = list(token_indices_set)

            if token_indices == []:
                statistics["num_no_token_indices"] += 1
                logger.error(f"There are no token indices for nugget '{nugget.text}' in '{context_sentence}'!")
                logger.error("==> Using all-zero embedding vector.")
                embedding: np.ndarray = np.zeros_like(output[0])
            else:
                # compute the embedding as the mean of the nugget's tokens' embeddings
                embedding: np.ndarray = np.mean(output[token_indices], axis=0)
            nugget[ContextSentenceEmbeddingSignal] = ContextSentenceEmbeddingSignal(embedding)

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "bert_resource_identifier": self._bert_resource_identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERTContextSentenceEmbedder":
        return cls(config["bert_resource_identifier"])


@register_configurable_element
class RelativePositionEmbedder(BaseEmbedder):
    """
    Position embedder that embeds the character position of a nugget relative to the start and end of the document.

    works on ASETNuggets:
    required signals: start_char and document.text
    produced signals: RelativePositionSignal
    """
    identifier: str = "RelativePositionEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [RelativePositionSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def __init__(self) -> None:
        super(RelativePositionEmbedder, self).__init__()
        logger.debug(f"Initialized '{self.identifier}'.")

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        for ix, nugget in enumerate(nuggets):
            self._use_status_callback_for_embedder(status_callback, "nuggets", ix, len(nuggets))
            if len(nugget.document.text) == 0:
                nugget[RelativePositionSignal] = RelativePositionSignal(0)
                statistics["num_text_is_empty"] += 1
            else:
                relative_position: float = nugget.start_char / len(nugget.document.text)
                nugget[RelativePositionSignal] = RelativePositionSignal(relative_position)

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RelativePositionEmbedder":
        return cls()


@register_configurable_element
class FastTextLabelEmbedder(BaseEmbedder):
    """
    Label embedder based on FastText.

    Splits the labels by '_' and by spaces and computes the embedding as the mean of the tokens' FastText embeddings.
    """
    identifier: str = "FastTextLabelEmbedder"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [NaturalLanguageLabelSignal.identifier],
        "attributes": [NaturalLanguageLabelSignal.identifier],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelEmbeddingSignal.identifier],
        "attributes": [LabelEmbeddingSignal.identifier],
        "documents": []
    }

    def __init__(self, embedding_resource_identifier: str, do_lowercase: bool, splitters: List[str]) -> None:
        """
        Initialize the FastTextLabelEmbedder.

        :param embedding_resource_identifier: identifier of the FastText resource
        :param do_lowercase: whether to lowercase tokens before embedding them
        :param splitters: characters at which the label should be split into tokens
        """
        super(FastTextLabelEmbedder, self).__init__()
        self._embedding_resource_identifier: str = embedding_resource_identifier
        self._do_lowercase: bool = do_lowercase
        self._splitters: List[str] = splitters

        # preload required resources
        resources.MANAGER.load(self._embedding_resource_identifier)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _compute_embedding(self, label: str, statistics: Statistics) -> LabelEmbeddingSignal:
        """
        Compute the embedding of the given label.

        :param label: given label to compute the embedding of
        :param statistics: statistics object to collect statistics
        :return: embedding signal of the label
        """
        # tokenize the label
        tokens: List[str] = [label]
        for splitter in self._splitters:
            new_tokens: List[str] = []
            for token in tokens:
                new_tokens += token.split(splitter)
            tokens: List[str] = new_tokens

        # lowercase the tokens
        if self._do_lowercase:
            tokens: List[str] = [token.lower() for token in tokens]

        # compute the embeddings
        embeddings: List[np.ndarray] = []
        for token in tokens:
            if token in resources.MANAGER[self._embedding_resource_identifier]:
                embeddings.append(resources.MANAGER[self._embedding_resource_identifier][token])
            else:
                statistics["num_unknown_tokens"] += 1
                statistics["unknown_tokens"].add(token)

        if embeddings == []:
            statistics["unable_to_embed_label"] += 1
            logger.error(f"Unable to embed label '{label}' with FastText, no known tokens!")
            assert False, f"Unable to embed label '{label}' with FastText, no known tokens!"
        else:
            # noinspection PyTypeChecker
            return LabelEmbeddingSignal(np.mean(np.array(embeddings), axis=0))

    def _embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["unknown_tokens"] = set()

        for ix, nugget in enumerate(nuggets):
            self._use_status_callback_for_embedder(status_callback, "nuggets", ix, len(nuggets))
            label: str = nugget[NaturalLanguageLabelSignal]
            nugget[LabelEmbeddingSignal] = self._compute_embedding(label, statistics)

    def _embed_attributes(
            self,
            attributes: List[ASETAttribute],
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        for ix, attribute in enumerate(attributes):
            self._use_status_callback_for_embedder(status_callback, "attributes", ix, len(attributes))
            label: str = attribute[NaturalLanguageLabelSignal]
            attribute[LabelEmbeddingSignal] = self._compute_embedding(label, statistics)

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "embedding_resource_identifier": self._embedding_resource_identifier,
            "do_lowercase": self._do_lowercase,
            "splitters": self._splitters
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastTextLabelEmbedder":
        return cls(config["embedding_resource_identifier"], config["do_lowercase"], config["splitters"])
