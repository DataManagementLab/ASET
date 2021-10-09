import abc
import logging
from typing import Optional, Callable, Dict, Any, List, Type

from spacy.tokens import Doc

from aset import resources
from aset.config import ConfigurableElement
from aset.data.annotations import SentenceStartCharsAnnotation
from aset.data.data import ASETDocumentBase, ASETNugget
from aset.data.signals import LabelSignal, POSTagsSignal

logger: logging.Logger = logging.getLogger(__name__)

EXTRACTORS: Dict[str, Type["BaseExtractor"]] = {}


def register_extractor(extractor: Type["BaseExtractor"]) -> Type["BaseExtractor"]:
    """Register the given extractor class."""
    EXTRACTORS[extractor.extractor_str] = extractor
    return extractor


class BaseExtractor(ConfigurableElement, abc.ABC):
    """
    Extractors obtain ASETNuggets from ASETDocuments.

    They are configurable elements and should be applied in the preprocessing phase. Each extractor comes with an
    identifier ('extractor_str').
    """
    extractor_str: str = "BaseExtractor"

    def __str__(self) -> str:
        return self.extractor_str

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.extractor_str == other.extractor_str

    @abc.abstractmethod
    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Obtain ASETNuggets from the documents of the given ASETDocumentBase.

        :param document_base: ASETDocumentBase to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseExtractor":
        return EXTRACTORS[config["extractor_str"]].from_config(config)

    def _use_status_fn(self, status_fn: Optional[Callable[[str, float], None]],
                       document_base: ASETDocumentBase, ix: int) -> None:
        """
        Helper method that calls the status function at regular intervals.

        :param status_fn: status function to call
        :param document_base: ASETDocumentBase to work on
        :param ix: index of the current document
        """
        if status_fn is not None:
            if ix == 0:
                status_fn(f"Running {self.extractor_str}...", 0)
            num_documents: int = len(document_base.documents)
            interval: int = num_documents // 10
            if interval != 0 and ix % interval == 0:
                status_fn(f"Running {self.extractor_str}...", ix / num_documents)


@register_extractor
class SpacyNERExtractor(BaseExtractor):
    """
    Extractor based on spacy's NER models.

    produced ASETDocument annotations: SentenceStartCharsAnnotation
    produced ASETNugget signals: LabelSignal, POSTagsSignal
    """
    extractor_str: str = "SpacyNERExtractor"

    def __init__(self, spacy_resource_str: str) -> None:
        """
        Initialize the SpacyNERExtractor.

        :param spacy_resource_str: identifier of the spacy model resource
        """
        super(SpacyNERExtractor, self).__init__()
        self._spacy_resource_str: str = spacy_resource_str

        # preload required resources
        resources.MANAGER.load(self._spacy_resource_str)
        logger.debug(f"Initialized {self.extractor_str}.")

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:

        if statistics is not None:
            statistics["num_documents"] = len(document_base.documents)
            statistics["num_nuggets"] = 0
            statistics["spacy_entity_type_dist"] = {}

        for ix, document in enumerate(document_base.documents):
            self._use_status_fn(status_fn, document_base, ix)

            spacy_output: Doc = resources.MANAGER[self._spacy_resource_str](document.text)

            sentence_start_chars: List[int] = []

            # transform the spacy output into the ASET document and nuggets
            for sentence in spacy_output.sents:
                sentence_start_chars.append(sentence.start_char)

            # noinspection PyTypeChecker
            annotation: SentenceStartCharsAnnotation = SentenceStartCharsAnnotation(sentence_start_chars)
            document.annotations[SentenceStartCharsAnnotation.annotation_str] = annotation

            for entity in spacy_output.ents:
                type_str = entity.label_  # TODO: create type mappings
                nugget: ASETNugget = ASETNugget(
                    document=document,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    extractor_str=self.extractor_str,
                    type_str=type_str,
                    value=None
                )

                nugget.signals[POSTagsSignal.signal_str] = POSTagsSignal([])  # TODO: gather pos tags

                nugget.signals[LabelSignal.signal_str] = LabelSignal(entity.label_)  # TODO: create label mappings
                document.nuggets.append(nugget)

                if statistics is not None:
                    statistics["num_nuggets"] += 1
                    statistics["spacy_entity_type_dist"][entity.label_] = \
                        statistics["spacy_entity_type_dist"].get(entity.label_, 0) + 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "extractor_str": self.extractor_str,
            "spacy_resource_str": self._spacy_resource_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpacyNERExtractor":
        return cls(config["spacy_resource_str"])


@register_extractor
class StanzaNERExtractor(BaseExtractor):
    """
    Extractor based on Stanza's NER model.

    produced ASETDocument annotations: SentenceStartCharsAnnotation
    produced ASETNugget signals: LabelSignal, POSTagsSignal
    """
    extractor_str: str = "StanzaNERExtractor"

    def __init__(self) -> None:
        """Initialize the StanzaNERExtractor."""
        super(StanzaNERExtractor, self).__init__()

        # preload required resources
        resources.MANAGER.load("StanzaNERPipeline")
        logger.debug(f"Initialized {self.extractor_str}.")

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:

        if statistics is not None:
            statistics["num_documents"] = len(document_base.documents)
            statistics["num_nuggets"] = 0
            statistics["stanza_entity_type_dist"] = {}

        for ix, document in enumerate(document_base.documents):
            self._use_status_fn(status_fn, document_base, ix)

            stanza_output = resources.MANAGER["StanzaNERPipeline"](document.text)

            sentence_start_chars: List[int] = []

            # transform the stanza output into the ASET document and nuggets
            for sentence in stanza_output.sentences:
                sentence_start_chars.append(sentence.tokens[0].start_char)

                for entity in sentence.entities:
                    type_str: str = {
                        "PERCENT": "number",
                        "QUANTITY": "number",
                        "ORDINAL": "number",
                        "CARDINAL": "number",
                        "MONEY": "number",
                        "DATE": "datetime",
                        "TIME": "datetime",
                        "PERSON": "string",
                        "NORP": "string",
                        "FAC": "string",
                        "ORG": "string",
                        "GPE": "string",
                        "LOC": "string",
                        "PRODUCT": "string",
                        "EVENT": "string",
                        "WORK_OF_ART": "string",
                        "LAW": "string",
                        "LANGUAGE": "string"
                    }[entity.type]

                    nugget: ASETNugget = ASETNugget(
                        document=document,
                        start_char=entity.start_char,
                        end_char=entity.start_char + len(entity.text),
                        extractor_str=self.extractor_str,
                        type_str=type_str,
                        value=None
                    )

                    pos_tags: List[str] = [word.xpos for word in entity.words]
                    nugget.signals[POSTagsSignal.signal_str] = POSTagsSignal(pos_tags)

                    label: str = {
                        "QUANTITY": "quantity measurement weight distance",
                        "CARDINAL": "cardinal numeral",
                        "NORP": "nationality religion political group",
                        "FAC": "building airport highway bridge",
                        "ORG": "organization",
                        "GPE": "country city state",
                        "LOC": "location mountain range body of water",
                        "PRODUCT": "product vehicle weapon food",
                        "EVENT": "event hurricane battle war sports",
                        "WORK_OF_ART": "work of art title of book song",
                        "LAW": "law document",
                        "LANGUAGE": "language",
                        "ORDINAL": "ordinal",
                        "MONEY": "money",
                        "PERCENT": "percentage",
                        "DATE": "date period",
                        "TIME": "time",
                        "PERSON": "person",
                    }[entity.type]

                    nugget.signals[LabelSignal.signal_str] = LabelSignal(label)
                    document.nuggets.append(nugget)

                    if statistics is not None:
                        statistics["num_nuggets"] += 1
                        statistics["stanza_entity_type_dist"][entity.type] = \
                            statistics["stanza_entity_type_dist"].get(entity.type, 0) + 1

            # noinspection PyTypeChecker
            annotation: SentenceStartCharsAnnotation = SentenceStartCharsAnnotation(sentence_start_chars)
            document.annotations[SentenceStartCharsAnnotation.annotation_str] = annotation

    def to_config(self) -> Dict[str, Any]:
        return {
            "extractor_str": self.extractor_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StanzaNERExtractor":
        return cls()
