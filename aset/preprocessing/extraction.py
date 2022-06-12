import abc
import json
import logging
from typing import Any, Dict, List

import requests
from spacy.tokens import Doc

from aset import resources
from aset.configuration import register_configurable_element, BasePipelineElement
from aset.data.data import ASETDocumentBase, ASETNugget
from aset.data.signals import LabelSignal, POSTagsSignal, SentenceStartCharsSignal
from aset.interaction import BaseInteractionCallback
from aset.resources import StanzaNERPipeline, FigerNERPipeline
from aset.statistics import Statistics
from aset.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class BaseExtractor(BasePipelineElement, abc.ABC):
    """
    Base class for all extractors.

    Extractors derive the information nuggets from the documents.
    """
    identifier: str = "BaseExtractor"


########################################################################################################################
# actual extractors
########################################################################################################################


@register_configurable_element
class SpacyNERExtractor(BaseExtractor):
    """Extractor based on spacy's NER models."""

    identifier: str = "SpacyNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self, spacy_resource_identifier: str) -> None:
        """
        Initialize the SpacyNERExtractor.

        :param spacy_resource_identifier: identifier of the spacy model resource
        """
        super(SpacyNERExtractor, self).__init__()
        self._spacy_resource_identifier: str = spacy_resource_identifier

        # preload required resources
        resources.MANAGER.load(self._spacy_resource_identifier)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: ASETDocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            spacy_output: Doc = resources.MANAGER[self._spacy_resource_identifier](document.text)
            sentence_start_chars: List[int] = []

            # transform the spacy output into the ASET document and nuggets
            for sentence in spacy_output.sents:
                sentence_start_chars.append(sentence.start_char)

            document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)

            for entity in spacy_output.ents:
                nugget: ASETNugget = ASETNugget(
                    document=document,
                    start_char=entity.start_char,
                    end_char=entity.end_char
                )

                nugget[POSTagsSignal] = POSTagsSignal([])  # TODO: gather pos tags
                nugget[LabelSignal] = LabelSignal(entity.label_)

                document.nuggets.append(nugget)

                statistics["num_nuggets"] += 1
                statistics["spacy_entity_type_dist"][entity.label_] += 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "spacy_resource_identifier": self._spacy_resource_identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpacyNERExtractor":
        return cls(config["spacy_resource_identifier"])


@register_configurable_element
class StanzaNERExtractor(BaseExtractor):
    """Extractor based on Stanza's NER model."""

    identifier: str = "StanzaNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self) -> None:
        """Initialize the StanzaNERExtractor."""
        super(StanzaNERExtractor, self).__init__()

        # preload required resources
        resources.MANAGER.load(StanzaNERPipeline)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: ASETDocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            stanza_output = resources.MANAGER[StanzaNERPipeline](document.text)

            sentence_start_chars: List[int] = []

            # transform the stanza output into the ASET document and nuggets
            for sentence in stanza_output.sentences:
                sentence_start_chars.append(sentence.tokens[0].start_char)

                for entity in sentence.entities:

                    nugget: ASETNugget = ASETNugget(
                        document=document,
                        start_char=entity.start_char,
                        end_char=entity.start_char + len(entity.text)
                    )

                    nugget[POSTagsSignal] = POSTagsSignal([word.xpos for word in entity.words])
                    nugget[LabelSignal] = LabelSignal(entity.type)

                    document.nuggets.append(nugget)

                    statistics["num_nuggets"] += 1
                    statistics["stanza_entity_type_dist"][entity.type] += 1

            document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StanzaNERExtractor":
        return cls()
