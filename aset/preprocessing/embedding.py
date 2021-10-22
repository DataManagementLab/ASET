import abc
import logging
import time
from typing import Optional, Dict, Any, List, Set, Type

import numpy as np

from aset import resources
from aset.config import ConfigurableElement
from aset.data.annotations import SentenceStartCharsAnnotation
from aset.data.data import ASETDocumentBase, ASETNugget, ASETAttribute
from aset.data.signals import LabelSignal, LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, \
    RelativePositionSignal, UserProvidedExamplesSignal
from aset.statistics import Statistics
from aset.status import StatusFunction

logger: logging.Logger = logging.getLogger(__name__)

EMBEDDERS: Dict[str, Type["BaseEmbedder"]] = {}


def register_embedder(embedder: Type["BaseEmbedder"]) -> Type["BaseEmbedder"]:
    """Register the given embedder class."""
    EMBEDDERS[embedder.embedder_str] = embedder
    return embedder


class BaseEmbedder(ConfigurableElement, abc.ABC):
    """
    Embedders work on ASETNuggets and ASETAttributes to transform signals and other information into embedding signals.

    They are configurable elements and should be applied in the preprocessing phase. Each embedder comes with an
    identifier ('embedder_str'). The embedders are free in which signals they require as inputs and which signals they
    produce as outputs. Furthermore, they do not have to work on both ASETNuggets and ASETAttributes
    """
    embedder_str: str = "BaseEmbedder"

    def __str__(self) -> str:
        return self.embedder_str

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.embedder_str == other.embedder_str

    def __hash__(self) -> int:
        return hash(self.embedder_str)

    def _use_status_fn(
            self,
            status_fn: StatusFunction,
            element: str,
            total: int,
            ix: int
    ) -> None:
        """
        Helper method that calls the status function at regular intervals.

        :param status_fn: status function to call
        :param element: 'nuggets' or 'attributes'
        :param total: total number of elements
        :param ix: index of the current elements
        """
        if total == 0:
            status_fn(f"Embedding {element} with {self.embedder_str}...", -1)
        elif ix == 0:
            status_fn(f"Embedding {element} with {self.embedder_str}...", 0)
        else:
            interval: int = total // 10
            if interval != 0 and ix % interval == 0:
                status_fn(f"Embedding {element} with {self.embedder_str}...", ix / total)

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Compute embedding signals for ASETNuggets and ASETAttributes of the given ASETDocumentBase.

        :param document_base: ASETDocumentBase to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        statistics["embedder_str"] = self.embedder_str

        # compute embeddings for the nuggets
        nuggets: List[ASETNugget] = document_base.nuggets
        logger.info(f"Embed {len(nuggets)} nuggets with '{self.embedder_str}'.")
        tick: float = time.time()
        status_fn(f"Embedding nuggets with {self.embedder_str}...", -1)
        self.embed_nuggets(nuggets, status_fn, statistics["nuggets"])
        status_fn(f"Embedding nuggets with {self.embedder_str}...", 1)
        tack: float = time.time()
        logger.info(f"Embedded {len(nuggets)} nuggets with '{self.embedder_str}' in {tack - tick} seconds.")

        # compute embeddings for the attributes
        attributes: List[ASETAttribute] = document_base.attributes
        logger.info(f"Embed {len(attributes)} attributes with '{self.embedder_str}'.")
        tick: float = time.time()
        status_fn(f"Embedding attributes with {self.embedder_str}...", -1)
        self.embed_attributes(attributes, status_fn, statistics["attributes"])
        status_fn(f"Embedding attributes with {self.embedder_str}...", 1)
        tack: float = time.time()
        logger.info(f"Embedded {len(attributes)} attributes with '{self.embedder_str}' in {tack - tick} seconds.")

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Compute embeddings for the given list of ASETNuggets.

        :param nuggets: list of ASETNuggets to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing

    def embed_attributes(
            self,
            attributes: List[ASETAttribute],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Compute embeddings for the given list of ASETNuggets.

        :param attributes: list of ASETAttributes to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseEmbedder":
        return EMBEDDERS[config["embedder_str"]].from_config(config)


class BaseSBERTEmbedder(BaseEmbedder, abc.ABC):
    """Base class for all embedders based on SBERT."""
    embedder_str: str = "BaseSBERTEmbedder"

    def __init__(self, sbert_resource_str: str) -> None:
        """
        Initialize the embedder.

        :param sbert_resource_str: identifier of the SBERT model resource
        """
        super(BaseSBERTEmbedder, self).__init__()
        self._sbert_resource_str: str = sbert_resource_str

        # preload required resources
        resources.MANAGER.load(self._sbert_resource_str)
        logger.debug(f"Initialized {self.embedder_str}.")

    def to_config(self) -> Dict[str, Any]:
        return {
            "embedder_str": self.embedder_str,
            "sbert_resource_str": self._sbert_resource_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseSBERTEmbedder":
        return cls(config["sbert_resource_str"])


@register_embedder
class SBERTLabelEmbedder(BaseSBERTEmbedder):
    """
    Label embedder based on SBERT.

    works on ASETNuggets:
    required signals: LabelSignal
    produced signals: LabelEmbeddingSignal

    works on ASETAttributes:
    required signals: name
    produced signals: LabelEmbeddingSignal
    """
    embedder_str: str = "SBERTLabelEmbedder"

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [nugget[LabelSignal] for nugget in nuggets]
        embeddings: List[np.array] = resources.MANAGER[self._sbert_resource_str].encode(texts, show_progress_bar=False)

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[LabelEmbeddingSignal] = LabelEmbeddingSignal(embedding)

    def embed_attributes(
            self,
            attributes: List[ASETAttribute],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [attribute.name for attribute in attributes]
        embeddings: List[np.array] = resources.MANAGER[self._sbert_resource_str].encode(texts, show_progress_bar=False)

        for attribute, embedding in zip(attributes, embeddings):
            attribute[LabelEmbeddingSignal] = LabelEmbeddingSignal(embedding)


@register_embedder
class SBERTTextEmbedder(BaseSBERTEmbedder):
    """
    Text embedder based on SBERT.

    works on ASETNuggets:
    required signals: text
    produced signals: TextEmbeddingSignal
    """
    embedder_str: str = "SBERTTextEmbedder"

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        texts: List[str] = [nugget.text for nugget in nuggets]
        embeddings: List[np.array] = resources.MANAGER[self._sbert_resource_str].encode(texts, show_progress_bar=False)

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[TextEmbeddingSignal] = TextEmbeddingSignal(embedding)


@register_embedder
class SBERTExamplesEmbedder(BaseSBERTEmbedder):
    """
    SBERT-based embedder to embed user-provided example texts.

    works on ASETAttributes:
    required signals: UserProvidedExamplesSignal
    produced signals: TextEmbeddingSignal
    """
    embedder_str: str = "SBERTExamplesEmbedder"

    def embed_attributes(
            self,
            attributes: List[ASETAttribute],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        for ix, attribute in enumerate(attributes):
            self._use_status_fn(status_fn, "attributes", len(attributes), ix)
            texts: List[str] = attribute[UserProvidedExamplesSignal]
            if texts != []:
                embeddings: List[np.array] = resources.MANAGER[self._sbert_resource_str].encode(
                    texts, show_progress_bar=False
                )
                embedding: np.array = np.mean(embeddings, axis=0)
                attribute[TextEmbeddingSignal] = TextEmbeddingSignal(embedding)
            else:
                statistics["num_no_examples"] += 1


@register_embedder
class SBERTContextSentenceEmbedder(BaseSBERTEmbedder):
    """
    Context sentence embedder based on SBERT.

    works on ASETNuggets:
    required signals: start_char, end_char, and document.text
    produced signals: ContextSentenceEmbeddingSignal
    """
    embedder_str: str = "SBERTContextSentenceEmbedder"

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        # collect context sentences
        texts: List[str] = []
        for nugget in nuggets:
            sent_start_chars: List[int] = nugget.document[SentenceStartCharsAnnotation]
            context_start_char: int = 0
            context_end_char: int = 0
            for ix, sent_start_char in enumerate(sent_start_chars):
                if sent_start_char > nugget.start_char:
                    if ix == 0:
                        context_start_char: int = 0
                        context_end_char: int = sent_start_char
                        break
                    else:
                        context_start_char: int = sent_start_chars[ix - 1]
                        context_end_char: int = sent_start_char
                        break
            else:
                if sent_start_chars != []:
                    context_start_char: int = sent_start_chars[-1]
                    context_end_char: int = len(nugget.document.text)
            texts.append(nugget.document.text[context_start_char:context_end_char])

        # compute embeddings
        embeddings: List[np.array] = resources.MANAGER[self._sbert_resource_str].encode(texts, show_progress_bar=False)

        for nugget, embedding in zip(nuggets, embeddings):
            nugget[ContextSentenceEmbeddingSignal] = ContextSentenceEmbeddingSignal(embedding)


@register_embedder
class BERTContextSentenceEmbedder(BaseEmbedder):
    """
    Context sentence embedder based on BERT.

    Computes the context embedding of an ASETNugget as the mean of the final hidden states of the tokens that make up
    the nugget in its context sentence.

    works on ASETNuggets:
    required signals: start_char, end_char, and document.text
    produced signals: ContextSentenceEmbeddingSignal
    """
    embedder_str: str = "BERTContextSentenceEmbedder"

    def __init__(self, bert_resource_str: str) -> None:
        """
        Initialize the BERTContextSentenceEmbedder.

        :param bert_resource_str: identifier of the BERT model resource
        """
        super(BERTContextSentenceEmbedder, self).__init__()
        self._bert_resource_str: str = bert_resource_str

        # preload required resources
        resources.MANAGER.load(self._bert_resource_str)
        logger.debug(f"Initialized {self.embedder_str}.")

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:

        for nugget_ix, nugget in enumerate(nuggets):
            self._use_status_fn(status_fn, "nuggets", len(nuggets), nugget_ix)
            # get the context sentence
            sent_start_chars: List[int] = nugget.document[SentenceStartCharsAnnotation]
            context_start_char: int = 0
            context_end_char: int = 0
            for ix, sent_start_char in enumerate(sent_start_chars):
                if sent_start_char > nugget.start_char:
                    if ix == 0:
                        context_start_char: int = 0
                        context_end_char: int = sent_start_char
                        break
                    else:
                        context_start_char: int = sent_start_chars[ix - 1]
                        context_end_char: int = sent_start_char
                        break
            else:
                if sent_start_chars != []:
                    context_start_char: int = sent_start_chars[-1]
                    context_end_char: int = len(nugget.document.text)

            context_sentence: str = nugget.document.text[context_start_char:context_end_char]
            start_in_context: int = nugget.start_char - context_start_char
            end_in_context: int = nugget.end_char - context_start_char

            # compute the sentence's token embeddings
            encoding = resources.MANAGER[self._bert_resource_str]["tokenizer"](
                context_sentence,
                return_tensors="pt",
                padding=True
            )

            device = resources.MANAGER[self._bert_resource_str]["device"]
            if device is not None:
                input_ids = encoding.input_ids.to(device)
                token_type_ids = encoding.token_type_ids.to(device)
                attention_mask = encoding.attention_mask.to(device)
            else:
                input_ids = encoding.input_ids
                token_type_ids = encoding.token_type_ids
                attention_mask = encoding.attention_mask

            outputs = resources.MANAGER[self._bert_resource_str]["model"](
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            output = outputs[0].detach()
            if device is not None:
                output = output.cpu()
            output: np.array = output[0].numpy()

            # determine which tokens make up the nugget
            token_indices: Set[int] = set()
            for char_ix in range(start_in_context, end_in_context):
                token_ix: Optional[int] = encoding.char_to_token(char_ix)
                if token_ix is not None:
                    token_indices.add(token_ix)
            token_indices: List[int] = list(token_indices)

            if token_indices == []:
                statistics["num_no_token_indices"] += 1
                logger.error(f"There are no token indices for nugget '{nugget.text}' in '{context_sentence}'!")
                assert False, f"There are no token indices for nugget '{nugget.text}' in '{context_sentence}'!"

            # compute the embedding as the mean of the nugget's tokens' embeddings
            embedding: np.array = np.mean(output[token_indices], axis=0)
            nugget[ContextSentenceEmbeddingSignal] = ContextSentenceEmbeddingSignal(embedding)

    def to_config(self) -> Dict[str, Any]:
        return {
            "embedder_str": self.embedder_str,
            "bert_resource_str": self._bert_resource_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERTContextSentenceEmbedder":
        return cls(config["bert_resource_str"])


@register_embedder
class RelativePositionEmbedder(BaseEmbedder):
    """
    Position embedder that embeds the character position of a nugget relative to the start and end of the document.

    works on ASETNuggets:
    required signals: start_char and document.text
    produced signals: RelativePositionSignal
    """
    embedder_str: str = "RelativePositionEmbedder"

    def __init__(self) -> None:
        super(RelativePositionEmbedder, self).__init__()
        logger.debug(f"Initialized {self.embedder_str}.")

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        for ix, nugget in enumerate(nuggets):
            self._use_status_fn(status_fn, "nuggets", len(nuggets), ix)
            if len(nugget.document.text) == 0:
                nugget[RelativePositionSignal] = RelativePositionSignal(0)
            else:
                relative_position: float = nugget.start_char / len(nugget.document.text)
                nugget[RelativePositionSignal] = RelativePositionSignal(relative_position)

    def to_config(self) -> Dict[str, Any]:
        return {
            "embedder_str": self.embedder_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RelativePositionEmbedder":
        return cls()


@register_embedder
class FastTextLabelEmbedder(BaseEmbedder):
    """
    Label embedder based on FastText.

    Splits the labels by '_' and by spaces and computes the embedding as the mean of the tokens' FastText embeddings.

    works on ASETNuggets:
    required signals: LabelSignal
    produced signals: LabelEmbeddingSignal

    works on ASETAttributes:
    required signals: name
    produced signals: LabelEmbeddingSignal
    """
    embedder_str: str = "FastTextLabelEmbedder"

    def __init__(self, embedding_resource_str: str, do_lowercase: bool, splitters: List[str]) -> None:
        """
        Initialize the FastTextLabelEmbedder.

        :param embedding_resource_str: identifier of the FastText resource
        :param do_lowercase: whether to lowercase tokens before embedding them
        :param splitters: characters at which the label should be split into tokens
        """
        super(FastTextLabelEmbedder, self).__init__()
        self._embedding_resource_str: str = embedding_resource_str
        self._do_lowercase: bool = do_lowercase
        self._splitters: List[str] = splitters

        # preload required resources
        resources.MANAGER.load(self._embedding_resource_str)
        logger.debug(f"Initialized {self.embedder_str}.")

    def _compute_embedding(
            self,
            label: str,
            statistics: Statistics
    ) -> LabelEmbeddingSignal:
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
        embeddings: List[np.array] = []
        for token in tokens:
            if token in resources.MANAGER[self._embedding_resource_str].keys():
                embeddings.append(resources.MANAGER[self._embedding_resource_str][token])
            statistics["num_unknown_tokens"] += 1

        if embeddings == []:
            statistics["unable_to_embed_label"] += 1
            logger.error(f"Unable to embed label '{label}' with FastText, no known tokens!")
            assert False, f"Unable to embed label '{label}' with FastText, no known tokens!"
        else:
            # noinspection PyTypeChecker
            return LabelEmbeddingSignal(np.mean(np.array(embeddings), axis=0))

    def embed_nuggets(
            self,
            nuggets: List[ASETNugget],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        for ix, nugget in enumerate(nuggets):
            self._use_status_fn(status_fn, "nuggets", len(nuggets), ix)
            label: str = nugget[LabelSignal]
            nugget[LabelEmbeddingSignal] = self._compute_embedding(label, statistics)

    def embed_attributes(
            self,
            attributes: List[ASETAttribute],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        for ix, attribute in enumerate(attributes):
            self._use_status_fn(status_fn, "attributes", len(attributes), ix)
            attribute[LabelEmbeddingSignal] = self._compute_embedding(attribute.name, statistics)

    def to_config(self) -> Dict[str, Any]:
        return {
            "embedder_str": self.embedder_str,
            "embedding_resource_str": self._embedding_resource_str,
            "do_lowercase": self._do_lowercase,
            "splitters": self._splitters
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastTextLabelEmbedder":
        return cls(config["embedding_resource_str"], config["do_lowercase"], config["splitters"])
