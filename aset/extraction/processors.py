"""Processors that determine the actual values of extractions."""
import logging
from abc import ABC, abstractmethod

from aset.core.resources import get_stanford_corenlp_pipeline
from aset.extraction.common import Extraction

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Facility that processes extractions and determines their actual values. Super class of all processors."""
    processor_str = "BaseProcessor"

    def __eq__(self, other):
        return self.processor_str == other.processor_str

    @abstractmethod
    def __call__(self, extractions: [Extraction]):
        """Determine the actual values of the given extractions."""
        raise NotImplementedError


class StanfordCoreNLPDateTimeProcessor(BaseProcessor):
    """Processor for datetime extractions built upon the Stanford CoreNLP library."""
    processor_str = "Stanford-CoreNLP-DateTime"

    def __init__(self):
        """Load the necessary infrastructure."""
        super(StanfordCoreNLPDateTimeProcessor, self).__init__()
        # preload the required resources
        get_stanford_corenlp_pipeline()

    def __call__(self, extractions: [Extraction]):
        """Determine the actual values of the given extractions."""

        for i, extraction in enumerate(extractions):
            if i % (len(extractions) // 5) == 0:
                logger.info(f"{self.processor_str}: {round(i / len(extractions) * 100)} percent done.")
            if extraction.extraction_type_str == Extraction.datetime_extraction_type_str:

                corenlp_result = get_stanford_corenlp_pipeline().annotate(extraction.mention)
                if len(corenlp_result.mentions) != 1:
                    logger.error(f"{self.processor_str}: {len(corenlp_result.mentions)} mentions instead of 1!")
                    # TODO: handle when one Stanza extraction leads to several Stanford CoreNLP values
                else:
                    extraction.value = corenlp_result.mentions[0].timex.value


class StanfordCoreNLPNumberProcessor(BaseProcessor):
    """Processor for numeric extractions built upon the Stanford CoreNLP library."""
    processor_str = "Stanford-CoreNLP-Number"

    def __init__(self):
        """Load the necessary infrastructure."""
        super(StanfordCoreNLPNumberProcessor, self).__init__()
        self.stanford_corenlp_pipeline = get_stanford_corenlp_pipeline()

    def __call__(self, extractions: [Extraction]):
        """Determine the actual values of the given extractions."""

        for i, extraction in enumerate(extractions):
            if i % (len(extractions) // 5) == 0:
                logger.info(f"{self.processor_str}: {round(i / len(extractions) * 100)} percent done.")
            if extraction.extraction_type_str == Extraction.number_extraction_type_str:

                corenlp_result = self.stanford_corenlp_pipeline.annotate(extraction.mention)
                if len(corenlp_result.mentions) != 1:
                    logger.error(f"{self.processor_str}: {len(corenlp_result.mentions)} mentions instead of 1!")
                    # TODO: handle when one Stanza extraction leads to several Stanford CoreNLP values
                else:
                    extraction.value = corenlp_result.mentions[0].normalizedNER


class StanfordCoreNLPStringProcessor(BaseProcessor):
    """Processor for string extractions built upon the Stanford CoreNLP library."""
    processor_str = "Stanford-CoreNLP-String"

    def __init__(self):
        """Load the necessary infrastructure."""
        super(StanfordCoreNLPStringProcessor, self).__init__()

    def __call__(self, extractions: [Extraction]):
        """Determine the actual values of the given extractions."""

        for i, extraction in enumerate(extractions):
            if i % (len(extractions) // 5) == 0:
                logger.info(f"{self.processor_str}: {round(i / len(extractions) * 100)} percent done.")
            if extraction.extraction_type_str == Extraction.string_extraction_type_str:
                extraction.value = extraction.mention  # TODO: implement canonicalization
